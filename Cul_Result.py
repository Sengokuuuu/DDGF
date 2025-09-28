import os
import sys
from pathlib import Path
import re
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from piqa.ssim import SSIM
from datetime import datetime
import time

def find_image_pairs(source_dir):
    source_path = Path(source_dir)
    
    if not source_path.exists():
        return []
    
    apy_folder = source_path / "Apy"
    if not apy_folder.exists() or not apy_folder.is_dir():
        return []
    
    generated_files = {}
    
    for file in source_path.iterdir():
        if file.is_file() and file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            file_name = file.name
            match = re.match(r'^(\d+)_0\.(png|jpg|jpeg)$', file_name, re.IGNORECASE)
            if match:
                identifier = match.group(1)
                generated_files[identifier] = file
    
    orig_files = {}
    
    for file in apy_folder.iterdir():
        if file.is_file() and file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            file_name = file.name
            match = re.match(r'^orig_(\d+)\.(png|jpg|jpeg)$', file_name, re.IGNORECASE)
            if match:
                identifier = match.group(1)
                orig_files[identifier] = file
    
    paired_files = []
    all_identifiers = set(generated_files.keys()) | set(orig_files.keys())
    
    for identifier in sorted(all_identifiers, key=lambda x: int(x)):
        if identifier in generated_files and identifier in orig_files:
            gen_file = generated_files[identifier]
            orig_file = orig_files[identifier]
            paired_files.append((gen_file, orig_file, identifier))
    
    return paired_files

class PSNR(nn.Module):
    def __init__(self, max_val=0):
        super().__init__()
        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()
        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)
    
    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)
        if mse == 0:
            return 0
        return 10 * torch.log10((1.0 / mse))

def calculate_nmse(img1, img2):
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    
    mse = torch.mean((img1_flat - img2_flat) ** 2)
    img2_mean = torch.mean(img2_flat)
    mse_norm = torch.mean((img2_flat - img2_mean) ** 2)
    
    if mse_norm == 0:
        return 0.0
    
    nmse = mse / mse_norm
    return nmse.item()

def calculate_uqi(img1, img2):
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    
    mu1 = torch.mean(img1_flat)
    mu2 = torch.mean(img2_flat)
    
    var1 = torch.var(img1_flat, unbiased=False)
    var2 = torch.var(img2_flat, unbiased=False)
    
    cov = torch.mean((img1_flat - mu1) * (img2_flat - mu2))
    
    numerator = 4 * cov * mu1 * mu2
    denominator = (var1 + var2) * (mu1**2 + mu2**2)
    
    if denominator == 0:
        return 1.0 if torch.equal(img1_flat, img2_flat) else 0.0
    
    uqi = numerator / denominator
    return uqi.item()

def calculate_metrics(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    
    if a.max() <= 1.0:
        a = a * 255.0
    if b.max() <= 1.0:
        b = b * 255.0
    
    if not a.flags.writeable:
        a = a.copy()
    if not b.flags.writeable:
        b = b.copy()
    
    if len(a.shape) == 2:
        a = np.stack([a, a, a], axis=2)
    elif len(a.shape) == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif len(a.shape) == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
        
    if len(b.shape) == 2:
        b = np.stack([b, b, b], axis=2)
    elif len(b.shape) == 3 and b.shape[2] == 1:
        b = np.repeat(b, 3, axis=2)
    elif len(b.shape) == 3 and b.shape[2] == 4:
        b = b[:, :, :3]
    
    if a.shape != b.shape:
        from PIL import Image
        if a.shape[:2] != b.shape[:2]:
            h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
            a = np.array(Image.fromarray(a.astype(np.uint8)).resize((w, h)))
            b = np.array(Image.fromarray(b.astype(np.uint8)).resize((w, h)))
        
        if a.shape[2] != b.shape[2]:
            if a.shape[2] == 1 and b.shape[2] == 3:
                a = np.repeat(a, 3, axis=2)
            elif a.shape[2] == 3 and b.shape[2] == 1:
                b = np.repeat(b, 3, axis=2)
    
    a, b = torch.from_numpy(a).float().cpu(), torch.from_numpy(b).float().cpu()
    
    if len(a.shape) == 3:
        a = a.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected image dimensions: {a.shape}")
        
    if len(b.shape) == 3:
        b = b.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"Unexpected image dimensions: {b.shape}")
    
    if a.shape[2] != b.shape[2] or a.shape[3] != b.shape[3]:
        a = nn.functional.interpolate(a, size=(b.shape[2], b.shape[3]))
    
    a_norm = a / 255.0
    b_norm = b / 255.0
    
    try:
        psnr = PSNR()(a_norm, b_norm).item()
        ssim = SSIM()(a_norm, b_norm).item()
        rmse = torch.sqrt(torch.mean((a.float() - b.float()) ** 2)).item()
        nmse = calculate_nmse(a_norm, b_norm)
        uqi = calculate_uqi(a_norm, b_norm)
        
        return psnr, ssim, rmse, nmse, uqi
        
    except Exception as e:
        return 0.0, 0.0, 999.0, 1.0, 0.0

def calculate_metrics_direct(paired_files, base_name="metrics"):
    if not paired_files:
        return None
    
    sumpsnr, sumssim, sumrmse, sumnmse, sumuqi = 0., 0., 0., 0., 0.
    N = 0
    failed_count = 0
    
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_file = f"{base_name}_{timestamp}.txt"
    f_results = open(results_file, 'w')
    
    for gen_file, orig_file, identifier in tqdm(paired_files, desc="Processing"):
        try:
            generated = plt.imread(str(gen_file))
            original = plt.imread(str(orig_file))
            
            if generated is None or original is None:
                failed_count += 1
                continue
                
            if len(generated.shape) == 0 or len(original.shape) == 0:
                failed_count += 1
                continue
            
            psnr, ssim, rmse, nmse, uqi = calculate_metrics(generated, original)
            
            f_results.write(f"{identifier}: PSNR={psnr:.4f}, SSIM={ssim:.4f}, RMSE={rmse:.4f}, NMSE={nmse:.4f}, UQI={uqi:.4f}\n")
            f_results.flush()
            
            sumpsnr += psnr
            sumssim += ssim
            sumrmse += rmse
            sumnmse += nmse
            sumuqi += uqi
            
            N += 1
            
        except Exception as e:
            failed_count += 1
            continue
    
    if N > 0:
        mpsnr = sumpsnr / N
        mssim = sumssim / N
        mrmse = sumrmse / N
        mnmse = sumnmse / N
        muqi = sumuqi / N
        
        print(f"Average PSNR: {mpsnr:.4f}")
        print(f"Average SSIM: {mssim:.4f}")
        print(f"Average RMSE: {mrmse:.4f}")
        print(f"Average NMSE: {mnmse:.4f}")
        print(f"Average UQI: {muqi:.4f}")
        print(f"Processed: {N}/{len(paired_files)} pairs")
        
        f_results.write(f"\nAVG: PSNR={mpsnr:.4f}, SSIM={mssim:.4f}, RMSE={mrmse:.4f}, NMSE={mnmse:.4f}, UQI={muqi:.4f}\n")
        f_results.write(f"Processed: {N}/{len(paired_files)} pairs\n")
    
    f_results.close()
    return results_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <source_folder_path>")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    paired_files = find_image_pairs(source_dir)
    
    if not paired_files:
        print("Error: No paired files found")
        sys.exit(1)
    
    source_path = Path(source_dir)
    folder_name = source_path.name
    safe_folder_name = re.sub(r'[<>:"/\\|?*]', '_', folder_name)
    
    try:
        metrics_file = calculate_metrics_direct(paired_files, safe_folder_name)
        print(f"Results saved to: {metrics_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()