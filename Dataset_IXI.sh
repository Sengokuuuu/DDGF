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
