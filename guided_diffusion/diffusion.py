import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path, download
from functions.DDGF import DDGF_diffusion, DDGF_plus_diffusion

import torchvision.utils as tvu

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, create_classifier, classifier_defaults, args_to_dict
import random

from scipy.linalg import orth


def get_gaussian_noisy_img(img, noise_level):
    return img + torch.randn_like(img).cuda() * noise_level

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out

def color2gray(x):
    coef=1/3
    x = x[:,0,:,:] * coef + x[:,1,:,:]*coef +  x[:,2,:,:]*coef
    return x.repeat(1,3,1,1)

def gray2color(x):
    x = x[:,0,:,:]
    coef=1/3
    base = coef**2 + coef**2 + coef**2
    return torch.stack((x*coef/base, x*coef/base, x*coef/base), 1)    



def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self, simplified, alpha_prior, lambda_prior, omega_prior):
        cls_fn = None
        if self.config.model.type == 'openai':
            config_dict = vars(self.config.model)
            model = create_model(**config_dict)
            if self.config.model.use_fp16:
                model.convert_to_fp16()
            if self.config.model.class_cond:
                ckpt = os.path.join(self.args.exp, 'logs/imagenet/%dx%d_diffusion.pt' % (
                self.config.data.image_size, self.config.data.image_size))
                if not os.path.exists(ckpt):
                    download(
                        'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/%dx%d_diffusion_uncond.pt' % (
                        self.config.data.image_size, self.config.data.image_size), ckpt)
            else:
                if self.config.data.dataset == 'IXI':
                    ckpt = os.path.join(self.args.exp, "logs/IXI/IXI.pt")

                elif self.config.data.dataset == 'PMUB':
                    ckpt = os.path.join(self.args.exp, "logs/PMUB/PMUB.pt")

                elif self.config.data.dataset == 'BraTS':
                    ckpt = os.path.join(self.args.exp, "logs/BraTS/BraTS.pt")

            model.load_state_dict(torch.load(ckpt, map_location=self.device))
            model.to(self.device)
            model.eval()
            model = torch.nn.DataParallel(model)

        if simplified:
            print('Run Simplified DDGF, without SVD.',
                  f'{self.config.time_travel.T_sampling} sampling steps.',
                  f'travel_length = {self.config.time_travel.travel_length},',
                  f'travel_repeat = {self.config.time_travel.travel_repeat}.',
                  f'Task: {self.args.deg}.'
                 )
            self.simplified_DDGF_plus(model, cls_fn)
        else:
            print('Run SVD-based DDGF.',
                  f'{self.config.time_travel.T_sampling} sampling steps.',
                  f'travel_length = {self.config.time_travel.travel_length},',
                  f'travel_repeat = {self.config.time_travel.travel_repeat}.',
                  f'Task: {self.args.deg}.'
                 )
            self.svd_based_DDGF_plus(model, cls_fn)
            
            
    def simplified_DDGF_plus(self, model, cls_fn):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(args, config)

        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # get degradation operator
        print("args.deg:",args.deg)
        if args.deg =='colorization':
            A = lambda z: color2gray(z)
            Ap = lambda z: gray2color(z)
        elif args.deg =='denoising':
            A = lambda z: z
            Ap = A

        elif args.deg =='sr_averagepooling':
            scale=round(args.deg_scale)
            A = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
            Ap = lambda z: MeanUpsample(z,scale)
        elif args.deg =='inpainting':
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device)
            A = lambda z: z*mask
            Ap = A
        elif args.deg =='mask_color_sr':
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device)
            A1 = lambda z: z*mask
            A1p = A1
            
            A2 = lambda z: color2gray(z)
            A2p = lambda z: gray2color(z)
            
            scale=round(args.deg_scale)
            A3 = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
            A3p = lambda z: MeanUpsample(z,scale)
            
            A = lambda z: A3(A2(A1(z)))
            Ap = lambda z: A1p(A2p(A3p(z)))
        elif args.deg =='diy':
            # design your own degradation
            loaded = np.load("exp/inp_masks/mask.npy")
            mask = torch.from_numpy(loaded).to(self.device)
            A1 = lambda z: z*mask
            A1p = A1
            
            A2 = lambda z: color2gray(z)
            A2p = lambda z: gray2color(z)
            
            scale=args.deg_scale
            A3 = torch.nn.AdaptiveAvgPool2d((256//scale,256//scale))
            A3p = lambda z: MeanUpsample(z,scale)
            
            A = lambda z: A3(A2(A1(z)))
            Ap = lambda z: A1p(A2p(A3p(z)))
        else:
            raise NotImplementedError("degradation type not supported")

        args.sigma_y = 2 * args.sigma_y #to account for scaling to [-1,1]
        sigma_y = args.sigma_y
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y = A(x_orig)

            if config.sampling.batch_size!=1:
                raise ValueError("please change the config file to set batch size as 1")

            Apy = Ap(y)

            os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
            for i in range(len(Apy)):
                tvu.save_image(
                    inverse_data_transform(config, Apy[i]),
                    os.path.join(self.args.image_folder, f"Apy/Apy_{idx_so_far + i}")
                )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]),
                    os.path.join(self.args.image_folder, f"Apy/orig_{idx_so_far + i}")
                )
                
            # init x_T
            x = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():
                skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
                n = x.size(0)
                x0_preds = []
                xs = [x]
                
                times = get_schedule_jump(config.time_travel.T_sampling, 
                                               config.time_travel.travel_length, 
                                               config.time_travel.travel_repeat,
                                              )
                time_pairs = list(zip(times[:-1], times[1:]))
                
                
                # reverse diffusion sampling
                for i, j in tqdm.tqdm(time_pairs):
                    i, j = i*skip, j*skip
                    if j<0: j=-1 

                    if j < i: # normal sampling 
                        t = (torch.ones(n) * i).to(x.device)
                        next_t = (torch.ones(n) * j).to(x.device)
                        at = compute_alpha(self.betas, t.long())
                        at_next = compute_alpha(self.betas, next_t.long())
                        sigma_t = (1 - at_next**2).sqrt()
                        xt = xs[-1].to('cuda')

                        et = model(xt, t)

                        if et.size(1) == 6:
                            et = et[:, :3]

                        # Eq. 12
                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                        # Eq. 19
                        if sigma_t >= at_next*sigma_y:
                            lambda_t = 1.
                            gamma_t = (sigma_t**2 - (at_next*sigma_y)**2).sqrt()
                        else:
                            lambda_t = (sigma_t)/(at_next*sigma_y)
                            gamma_t = 0.

                        # Eq. 17
                        x0_t_hat = x0_t - lambda_t*Ap(A(x0_t) - y)

                        eta = self.args.eta

                        c1 = (1 - at_next).sqrt() * eta
                        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

                        # different from the paper, we use DDIM here instead of DDPM
                        xt_next = at_next.sqrt() * x0_t_hat + gamma_t * (c1 * torch.randn_like(x0_t) + c2 * et)

                        x0_preds.append(x0_t.to('cpu'))
                        xs.append(xt_next.to('cpu'))    
                    else: # time-travel back
                        next_t = (torch.ones(n) * j).to(x.device)
                        at_next = compute_alpha(self.betas, next_t.long())
                        x0_t = x0_preds[-1].to('cuda')

                        xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                        xs.append(xt_next.to('cpu'))

                x = xs[-1]
                
            x = [inverse_data_transform(config, xi) for xi in x]

            tvu.save_image(
                x[0], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{0}.png")
            )
            orig = inverse_data_transform(config, x_orig[0])
            mse = torch.mean((x[0].to(self.device) - orig) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            avg_psnr += psnr

            idx_so_far += y.shape[0]

            pbar.set_description("PSNR: %.2f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)

    def svd_based_DDGF_plus(self, model, cls_fn):
        args, config = self.args, self.config

        dataset, test_dataset = get_dataset(args, config)

        device_count = torch.cuda.device_count()

        if args.subset_start >= 0 and args.subset_end > 0:
            assert args.subset_end > args.subset_start
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.subset_start, args.subset_end))
        else:
            args.subset_start = 0
            args.subset_end = len(test_dataset)

        print(f'Dataset has size {len(test_dataset)}')

        def seed_worker(worker_id):
            worker_seed = args.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = data.DataLoader(
            test_dataset,
            batch_size=config.sampling.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
        )

        deg = args.deg
        A_funcs = None
        if deg == 'cs_walshhadamard':
            compress_by = round(1/args.deg_scale)
            from functions.svd_operators import WalshHadamardCS
            A_funcs = WalshHadamardCS(config.data.channels, self.config.data.image_size, compress_by,
                                      torch.randperm(self.config.data.image_size ** 2, device=self.device), self.device)
        elif deg == 'sr_averagepooling':
            blur_by = int(args.deg_scale)
            from functions.svd_operators import SuperResolution
            A_funcs = SuperResolution(config.data.channels, config.data.image_size, blur_by, self.device)
        elif deg == 'sr_bicubic':
            factor = int(args.deg_scale)
            from functions.svd_operators import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            A_funcs = SRConv(kernel / kernel.sum(), \
                             config.data.channels, self.config.data.image_size, self.device, stride=factor)
        else:
            raise ValueError("degradation type not supported")
        args.sigma_y = 2 * args.sigma_y
        sigma_y = args.sigma_y
        
        print(f'Start from {args.subset_start}')
        idx_init = args.subset_start
        idx_so_far = args.subset_start
        avg_psnr = 0.0
        pbar = tqdm.tqdm(val_loader)
        for x_orig, classes, img_names in pbar:
            x_orig = x_orig.to(self.device)
            x_orig = data_transform(self.config, x_orig)

            y = A_funcs.A(x_orig)
            
            b, hwc = y.size()
            if 'color' in deg:
                hw = hwc / 1
                h = w = int(hw ** 0.5)
                y = y.reshape((b, 1, h, w))
            elif 'inp' in deg or 'cs' in deg:
                pass
            else:
                hw = hwc / 3
                h = w = int(hw ** 0.5)
                y = y.reshape((b, 3, h, w))
                
            if self.args.add_noise: # for denoising test
                y = get_gaussian_noisy_img(y, sigma_y) 
            
            y = y.reshape((b, hwc))

            Apy = A_funcs.A_pinv(y).view(y.shape[0], config.data.channels, self.config.data.image_size,
                                                self.config.data.image_size)

            if deg[:6] == 'deblur':
                Apy = y.view(y.shape[0], config.data.channels, self.config.data.image_size,
                                    self.config.data.image_size)
            elif deg == 'colorization':
                Apy = y.view(y.shape[0], 1, self.config.data.image_size, self.config.data.image_size).repeat(1,3,1,1)
            elif deg == 'inpainting':
                Apy += A_funcs.A_pinv(A_funcs.A(torch.ones_like(Apy))).reshape(*Apy.shape) - 1

            os.makedirs(os.path.join(self.args.image_folder, "Apy"), exist_ok=True)
            for i in range(len(Apy)):
                save_name = img_names[i]
                tvu.save_image(
                    inverse_data_transform(config, Apy[i]),
                    os.path.join(self.args.image_folder, f"Apy/Apy_{save_name}.png")
                )
                tvu.save_image(
                    inverse_data_transform(config, x_orig[i]),
                    os.path.join(self.args.image_folder, f"Apy/orig_{save_name}.png")
                )

            #Start DDIM
            x = torch.randn(
                y.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )

            with torch.no_grad():
                if sigma_y==0.: 
                    x, _ = DDGF_diffusion(x, model, self.betas, self.args.eta, A_funcs, y, cls_fn=cls_fn, classes=classes, config=config, alpha_prior=args.alpha_prior, lambda_prior=args.lambda_prior, omega_prior=args.omega_prior)
                else: 
                    x, _ = DDGF_plus_diffusion(x, model, self.betas, self.args.eta, A_funcs, y, sigma_y, cls_fn=cls_fn, classes=classes, config=config, alpha_prior=args.alpha_prior, lambda_prior=args.lambda_prior, omega_prior=args.omega_prior )
            x = [inverse_data_transform(config, xi) for xi in x]

            if deg == 'sr_bicubic':
                for j in range(x[0].size(0)):
                    save_name = img_names[j] 
                    tvu.save_image(
                        x[0][j], os.path.join(self.args.image_folder, save_name)
                    )
                orig = inverse_data_transform(config, x_orig[j])
                mse = torch.mean((x[0][j].to(self.device) - orig) ** 2)
                psnr = 10 * torch.log10(1 / mse)
                avg_psnr += psnr
            else:
                for j in range(x[0].size(0)):
                    tvu.save_image(
                        x[0][j], os.path.join(self.args.image_folder, f"{idx_so_far + j}_{0}.png")
                    )
                orig = inverse_data_transform(config, x_orig[j])
                mse = torch.mean((x[0][j].to(self.device) - orig) ** 2)
                psnr = 10 * torch.log10(1 / mse)
                avg_psnr += psnr

            idx_so_far += y.shape[0]

            pbar.set_description("PSNR: %.4f" % (avg_psnr / (idx_so_far - idx_init)))

        avg_psnr = avg_psnr / (idx_so_far - idx_init)
        print("Total Average PSNR: %.4f" % avg_psnr)
        print("Number of samples: %d" % (idx_so_far - idx_init))

# Code form RePaint   
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    assert times[0] > times[1], (times[0], times[1])

    assert times[-1] == -1, times[-1]

    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
