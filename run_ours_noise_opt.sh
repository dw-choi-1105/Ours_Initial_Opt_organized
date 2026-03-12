# fireflow version
if [ $1 == 0 ]; then
    # FFHQ_DIR="/home/qkdwnstj10/ICML26/Datasets/FFHQ_10/00004.png"
    FFHQ_DIR="/root/data/FFHQ_5"
    python solve_ours_noise_opt_fireflow.py \
        --img_size 768 \
        --img_path ${FFHQ_DIR} \
        --workdir workdir_Ours_results_260311_test_fireflow/SR/FFHQ \
        --base_workdir workdir_Ours_results_260311_test_fireflow/SR/FFHQ \
        --prompt "a high quality photo of a face" \
        --method naive \
        --task deblur_motion \
        --operator_imp FFT \
        --deg_scale 61 \
        --noise_std 0.03 \
        --cfg_scale 1.0 \
        --seed 42 \
        --NFE 28 \
        --inner_NFE 3 \
        --n_experts 8 \
        --n_reflections_per_expert 4 \
        --noise_opt_steps 50 \
        --lr_noise_opt 0.001 \
        --lambda_jb 0 \
        --lambda_ks 0 \
        --lambda_orth 0 \
        --use_grad_checkpoint \
        --use_amp \
        --efficient_memory;
fi

# euler version
if [ $1 == 1 ]; then
    # FFHQ_DIR="/home/qkdwnstj10/ICML26/Datasets/FFHQ_10/00004.png"
    FFHQ_DIR="/root/data/FFHQ_5"
    python solve_ours_noise_opt_euler.py \
        --img_size 768 \
        --img_path ${FFHQ_DIR} \
        --workdir workdir_Ours_results_260311_test_euler/SR/FFHQ \
        --base_workdir workdir_Ours_results_260311_test_euler/SR/FFHQ \
        --prompt "a high quality photo of a face" \
        --method naive \
        --task deblur_motion \
        --operator_imp FFT \
        --deg_scale 61 \
        --noise_std 0.03 \
        --cfg_scale 1.0 \
        --seed 42 \
        --NFE 28 \
        --inner_NFE 3 \
        --n_experts 8 \
        --n_reflections_per_expert 4 \
        --noise_opt_steps 50 \
        --lr_noise_opt 0.001 \
        --lambda_jb 0 \
        --lambda_ks 0 \
        --lambda_orth 0 \
        --use_grad_checkpoint \
        --use_amp \
        --efficient_memory;
fi
