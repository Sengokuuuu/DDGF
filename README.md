# Zero-Shot Medical Image Super-Resolution Using Denoising Diffusion Models with Gradient-Frequency Priors

![](https://img.shields.io/badge/Python-3.10-blue.svg)  ![](https://img.shields.io/badge/PyTorch-2.7.0+-ee4c2c.svg)  ![](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Abstract

Diffusion models have demonstrated exceptional performance in image super-resolution tasks. However, in the field of medical imaging, they face unique challenges, including high noise sensitivity to scanning artifacts, loss of critical anatomical edges (such as tumor boundaries or vascular structures), and detail degradation due to frequency imbalances. To address these issues, we propose Denoising Diffusion Models with Gradient-Frequency Priors (DDGF), a novel zero-shot multi-prior driven framework for Medical Image Super-Resolution (MISR). This framework is built on the diffusion prior by integrating gradient and frequency-domain priors. Specifically, the gradient prior employs structure tensor-weighted fractional-order gradient regularization to promote edge preservation, noise suppression, and structural detail recovery, which is particularly suitable for noise-prone MRI scans. The frequency-domain prior is implemented via Gaussian low-pass filtering, emphasizing low-frequency structural integrity, thus suppressing high-frequency noise and artifacts, thus enhancing overall robustness in noisy environments. It complements the gradient prior to achieving balanced detail preservation. To validate the effectiveness of the proposed method, we conducted extensive experiments on multiple medical image super-resolution tasks using four medical image datasets. We compared it with state-of-the-art diffusion-based image restoration algorithms. Qualitative and quantitative results demonstrate its significant advantages. Ablation studies further confirm the superiority of the gradient prior in structural preservation and the efficacy of the frequency-domain prior in noise mitigation, thereby highlighting their synergistic contributions to clinical applications.

---

## 🎇 Architecture of DDGF

<img width="7370" height="3581" alt="Fig2_2_01" src="https://github.com/user-attachments/assets/57afe2e7-9e93-4ca2-b692-f484bc134d52" />

---

## 💡 Key Features

+ We propose a novel zero-shot MISR framework that optimizes multiple priors, including diffusion, gradient, and frequency-domain priors, enabling adaptation to diverse intra-modality restoration scenarios without retraining.
+ We propose a fractional-order gradient regularization prior, which is combined with structure tensors to achieve edge preservation, noise suppression, and structural detail recovery in MISR.
+ We propose a frequency-domain prior based on Gaussian low-pass filtering to suppress high-frequency artifacts while preserving low-frequency anatomical structures. It improves overall image quality and robustness in high-noise environments and complements the gradient prior for better detail balance.
+ Quantitative and qualitative experiments on multiple medical datasets demonstrate the effectiveness and synergistic effects of the proposed priors, with overall results outperforming existing state-of-the-art methods.

---

## <font style="color:rgb(51, 51, 51);">🚀</font><font style="color:rgb(51, 51, 51);"> </font><font style="color:#000000;background-color:#FFFFFF;">Installation && Usage</font>

### Code

```git
git@github.com:Sengokuuuu/DDGF.git
```

### <font style="color:#000000;background-color:#FFFFFF;">Environment</font>

```python
pip install numpy torch blobfile tqdm pyYaml pillow
```

### <font style="color:#000000;background-color:#FFFFFF;">Pre-Trained Models</font>

 For the pre-trained models of PMUB dataset, IXI dataset, and BraTS dataset, you can place them respectively in :

```plain
exp
├── logs
│   ├── PMUB
│     ├── PMUB.pt
│   ├── IXI
│     ├── IXI.pt
|   ├── BraTS
│     ├── BraTS.pt
```

> **Note:** Pre-trained models will be released upon paper acceptance.  

### Datasets

 For these three datasets, you can place them in :

```plain
exp
├── datasets
│   ├── PMUB
│     ├── PMUB
│   ├── IXI
│     ├── IXI
|   ├── BraTS
│     ├── BraTS
```

> **Note:** Dataset details and download links will be provided upon paper acceptance.  

### Test

 All testing scripts are provided in `Dataset_PMUB.sh`, `Dataset_IXI.sh`, and `Dataset_BraTS.sh`.  

 You can select different tasks for the corresponding datasets. For example:  

```plain
For PMUB Dataset
#####  noise-free tasks  #####

# DDGF_PMUB_4SR_Bicubic
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config PMUB.yml --path_y PMUB --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i DDGF_PMUB_4SR_B --lambda_prior 0.25 --alpha_prior 0.85 --omega_prior 0.5

# DDGF_PMUB_4SR_Averagepooling
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config PMUB.yml --path_y PMUB --eta 0.85 --deg "sr_averagepooling" --deg_scale 4 --sigma_y 0. -i DDGF_PMUB_4SR_A --lambda_prior 0.25 --alpha_prior 0.85 --omega_prior 0.5

# DDGF_PMUB_8SR_Bicubic
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config PMUB.yml --path_y PMUB --eta 0.85 --deg "sr_bicubic" --deg_scale 8 --sigma_y 0. -i DDGF_PMUB_8SR_B --lambda_prior 0.25 --alpha_prior 0.85 --omega_prior 0.5

# DDGF_PMUB_8SR_Averagepooling
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config PMUB.yml --path_y PMUB --eta 0.85 --deg "sr_averagepooling" --deg_scale 8 --sigma_y 0. -i DDGF_PMUB_8SR_A --lambda_prior 0.25 --alpha_prior 0.85 --omega_prior 0.5

# DDGF_PMUB_16SR_Bicubic
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config PMUB.yml --path_y PMUB --eta 0.85 --deg "sr_bicubic" --deg_scale 16 --sigma_y 0. -i DDGF_PMUB_16SR_B --lambda_prior 0.25 --alpha_prior 0.85 --omega_prior 0.5

# DDGF_PMUB_16SR_Averagepooling
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config PMUB.yml --path_y PMUB --eta 0.85 --deg "sr_averagepooling" --deg_scale 16 --sigma_y 0. -i DDGF_PMUB_16SR_A --lambda_prior 0.25 --alpha_prior 0.85 --omega_prior 0.5

#####  noise tasks  #####

# DDGF_PMUB_4SR_Bicubic_0.05
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config PMUB.yml --path_y PMUB --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0.05 -i DDGF_PMUB_4SR_B_0.05 --add_noise --lambda_prior 0.25 --alpha_prior 0.85 --omega_prior 0.5

# DDGF_PMUB_4SR_Averagepooling_0.05
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config PMUB.yml --path_y PMUB --eta 0.85 --deg "sr_averagepooling" --deg_scale 4 --sigma_y 0.05 -i DDGF_PMUB_4SR_A_0.05 --add_noise --lambda_prior 0.25 --alpha_prior 0.85 --omega_prior 0.5

# DDGF_PMUB_8SR_Bicubic_0.05
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config PMUB.yml --path_y PMUB --eta 0.85 --deg "sr_bicubic" --deg_scale 8 --sigma_y 0.05 -i DDGF_PMUB_8SR_B_0.05 --add_noise --lambda_prior 0.25 --alpha_prior 0.85 --omega_prior 0.5

# DDGF_PMUB_8SR_Averagepooling_0.05
CUDA_VISIBLE_DEVICES=0 python main.py --ni --config PMUB.yml --path_y PMUB --eta 0.85 --deg "sr_averagepooling" --deg_scale 8 --sigma_y 0.05 -i DDGF_PMUB_8SR_A_0.05 --add_noise --lambda_prior 0.25 --alpha_prior 0.85 --omega_prior 0.5

```

```plain
For IXI Dataset
##### noisy-free tasks #####

# DDGF_IXI_4SR_Bicubic
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config IXI.yml --path_y IXI --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i DDGF_IXI_4SR_B --lambda_prior 0.32 --alpha_prior 1.10 --omega_prior 0.5

# DDGF_IXI_4SR_Averagepooling
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config IXI.yml --path_y IXI --eta 0.85 --deg "sr_averagepooling" --deg_scale 4 --sigma_y 0. -i DDGF_IXI_4SR_A --lambda_prior 0.32 --alpha_prior 1.10 --omega_prior 0.5

# DDGF_IXI_8SR_Bicubic
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config IXI.yml --path_y IXI --eta 0.85 --deg "sr_bicubic" --deg_scale 8 --sigma_y 0. -i DDGF_IXI_8SR_B --lambda_prior 0.32 --alpha_prior 1.10 --omega_prior 0.5

# DDGF_IXI_8SR_Averagepooling
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config IXI.yml --path_y IXI --eta 0.85 --deg "sr_averagepooling" --deg_scale 8 --sigma_y 0. -i DDGF_IXI_8SR_A --lambda_prior 0.32 --alpha_prior 1.10 --omega_prior 0.5

# DDGF_IXI_16SR_Bicubic
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config IXI.yml --path_y IXI --eta 0.85 --deg "sr_bicubic" --deg_scale 16 --sigma_y 0. -i DDGF_IXI_16SR_B --lambda_prior 0.32 --alpha_prior 1.10 --omega_prior 0.5

# DDGF_IXI_16SR_Averagepooling
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config IXI.yml --path_y IXI --eta 0.85 --deg "sr_averagepooling" --deg_scale 16 --sigma_y 0. -i DDGF_IXI_16SR_A --lambda_prior 0.32 --alpha_prior 1.10 --omega_prior 0.5

##### noisy tasks #####

# DDGF_IXI_4SR_Bicubic_0.05
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config IXI.yml --path_y IXI --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0.05 -i DDGF_IXI_4SR_B_0.05 --add_noise --lambda_prior 0.32 --alpha_prior 1.10 --omega_prior 0.5

# DDGF_IXI_4SR_Averagepooling_0.05
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config IXI.yml --path_y IXI --eta 0.85 --deg "sr_averagepooling" --deg_scale 4 --sigma_y 0.05 -i DDGF_IXI_4SR_A_0.05 --add_noise --lambda_prior 0.32 --alpha_prior 1.10 --omega_prior 0.5

# DDGF_IXI_8SR_Bicubic_0.05
CUDA_VISIBLE_DEVICES=2 python main.py --ni --config IXI.yml --path_y IXI --eta 0.85 --deg "sr_bicubic" --deg_scale 8 --sigma_y 0.05 -i DDGF_IXI_8SR_B_0.05 --add_noise --lambda_prior 0.32 --alpha_prior 1.10 --omega_prior 0.5

# DDGF_IXI_8SR_Averagepooling_0.05
CUDA_VISIBLE_DEVICES= python main.py --ni --config IXI.yml --path_y IXI --eta 0.85 --deg "sr_averagepooling" --deg_scale 8 --sigma_y 0.05 -i DDGF_IXI_8SR_A_0.05 --add_noise --lambda_prior 0.32 --alpha_prior 1.10 --omega_prior 0.5

```

```plain
For BraTS Dataset
##### noisy-free tasks #####

# DDGF_BraTS_4SR_Bicubic
CUDA_VISIBLE_DEVICES=1 python main.py --ni --config BraTS.yml --path_y BraTS --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i DDGF_BraTS_4SR_B --lambda_prior 0.3 --alpha_prior 0.7 --omega_prior 0.5

# DDGF_BraTS_4SR_Averagepooling
CUDA_VISIBLE_DEVICES=1 python main.py --ni --config BraTS.yml --path_y BraTS --eta 0.85 --deg "sr_averagepooling" --deg_scale 4 --sigma_y 0. -i DDGF_BraTS_4SR_A --lambda_prior 0.3 --alpha_prior 0.7 --omega_prior 0.5

# DDGF_BraTS_8SR_Bicubic
CUDA_VISIBLE_DEVICES=1 python main.py --ni --config BraTS.yml --path_y BraTS --eta 0.85 --deg "sr_bicubic" --deg_scale 8 --sigma_y 0. -i DDGF_BraTS_8SR_B --lambda_prior 0.3 --alpha_prior 0.7 --omega_prior 0.5

# DDGF_BraTS_8SR_Averagepooling
CUDA_VISIBLE_DEVICES=1 python main.py --ni --config BraTS.yml --path_y BraTS --eta 0.85 --deg "sr_averagepooling" --deg_scale 8 --sigma_y 0. -i DDGF_BraTS_8SR_A --lambda_prior 0.3 --alpha_prior 0.7 --omega_prior 0.5

# DDGF_BraTS_16SR_Bicubic
CUDA_VISIBLE_DEVICES=1 python main.py --ni --config BraTS.yml --path_y BraTS --eta 0.85 --deg "sr_bicubic" --deg_scale 16 --sigma_y 0. -i DDGF_BraTS_16SR_B --lambda_prior 0.3 --alpha_prior 0.7 --omega_prior 0.5

# DDGF_BraTS_16SR_Averagepooling
CUDA_VISIBLE_DEVICES=1 python main.py --ni --config BraTS.yml --path_y BraTS --eta 0.85 --deg "sr_averagepooling" --deg_scale 16 --sigma_y 0. -i DDGF_BraTS_16SR_A --lambda_prior 0.3 --alpha_prior 0.7 --omega_prior 0.5

##### noisy tasks #####

# DDGF_BraTS_4SR_Bicubic_0.05
CUDA_VISIBLE_DEVICES=1 python main.py --ni --config BraTS.yml --path_y BraTS --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0.05 -i DDGF_BraTS_4SR_B_0.05 --add_noise --lambda_prior 0.3 --alpha_prior 0.7 --omega_prior 0.5

# DDGF_BraTS_4SR_Averagepooling_0.05
CUDA_VISIBLE_DEVICES=1 python main.py --ni --config BraTS.yml --path_y BraTS --eta 0.85 --deg "sr_averagepooling" --deg_scale 4 --sigma_y 0.05 -i DDGF_BraTS_4SR_A_0.05 --add_noise --lambda_prior 0.3 --alpha_prior 0.7 --omega_prior 0.5

# DDGF_BraTS_8SR_Bicubic_0.05
CUDA_VISIBLE_DEVICES=1 python main.py --ni --config BraTS.yml --path_y BraTS --eta 0.85 --deg "sr_bicubic" --deg_scale 8 --sigma_y 0.05 -i DDGF_BraTS_8SR_B_0.05 --add_noise --lambda_prior 0.3 --alpha_prior 0.7 --omega_prior 0.5

# DDGF_BraTS_8SR_Averagepooling_0.05
CUDA_VISIBLE_DEVICES=1 python main.py --ni --config BraTS.yml --path_y BraTS --eta 0.85 --deg "sr_averagepooling" --deg_scale 8 --sigma_y 0.05 -i DDGF_BraTS_8SR_A_0.05 --add_noise --lambda_prior 0.3 --alpha_prior 0.7 --omega_prior 0.5

```

### Cul Metrics

 To evaluate PSNR, SSIM, RMSE, UQI, and NMSE metrics, use the following command:  

```plain
python Cul_Result.py [path your results]
```

---

## 🎈 **Acknowledgements**

**We sincerely thank everyone for their tremendous contributions to this project, and extend our gratitude to the editors and reviewers for their dedicated work!**

