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
