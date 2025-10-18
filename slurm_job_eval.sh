#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=resgpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cp524

export HF_HOME="/vol/bitbucket/cp524/hf_cache"

# for offline loading only
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# Activate virtual environment
export PATH=/vol/bitbucket/cp524/dev/SMC_Meissonic_2/venv/bin:$PATH
source /vol/bitbucket/cp524/dev/SMC_Meissonic_2/venv/bin/activate

# Set up CUDA
source /vol/cuda/12.5.0/setup.sh

# Navigate to script directory
cd /vol/bitbucket/cp524/dev/SMC_Meissonic_2
export PYTHONPATH=$PYTHONPATH:$(pwd)

export PYTHONUNBUFFERED=1

# Run training notebook
MODEL="meissonic-fp16-monetico"
COMMON_ARGS=(
  --model_name="$MODEL"
  --guidance_reward_fn=ImageReward
  --metrics_to_compute=ImageReward#HumanPreference
  --resample_frequency=10
  --num_inference_steps=100
  --reward_name=image_reward_plus_5
  --lambda_tempering
  --lambda_one_at=100
  --continuous_formulation
  --CFG=9.0
  # --proposal_type=locally_optimal
  # --phi=4
  --tau=1.0
  --ess_threshold=0.5
  --partial_resampling
  --kl_weight=0.02
)


# Best-of-N
# for n in 1 2 4 8 16; do
#   echo "▶ Running Best-of-${n} with batch size=1"
#   python src/eval/launch_eval_runs_meissonic_ft.py \
#     "${COMMON_ARGS[@]}" \
#     --num_particles=$n \
#     --batch_p=$n \
#     --proposal_type="without_SMC"
# done


# Best-of-N (FT)
for n in 1 2 4 8 16; do
  echo "▶ Running Best-of-${n} (FT)"
  python src/eval/launch_eval_runs_meissonic_ft.py \
    "${COMMON_ARGS[@]}" \
    --num_particles=$n \
    --batch_p=$n \
    --proposal_type="without_SMC" \
    --lora_ckpt_uuid="9e7bb4ef-4aa8-43c5-8677-f6636098972c" \
    --reward_name="hps" \
    --guidance_reward_fn="HumanPreference"
done


# # SMC-Grad
# for phi in 1 4; do
#   for n in 1 2 4 8 16; do
#     if [ "$phi" -eq 1 ]; then
#       if [ "$n" -lt 8 ]; then
#           batch_size=$n
#       else
#           batch_size=8
#       fi
#     elif [ "$phi" -eq 4 ]; then
#         if [ "$n" -lt 4 ]; then
#             batch_size=$n
#         else
#             batch_size=4
#         fi
#     fi
#     echo "▶ Running SMC-Grad with phi = ${phi}, and batch size=${batch_size}"
#     python src/eval/launch_eval_runs_meissonic_ft.py \
#       "${COMMON_ARGS[@]}" \
#       --num_particles=8 \
#       --batch_p=$batch_size \
#       --proposal_type="locally_optimal" \
#       --phi=$phi
#   done
# done


# SMC (FT)
for phi in 1; do
  for n in 2 4 8 16; do
    echo "▶ Running SMC (FT) with phi = ${phi}, and n=${n}"
    python src/eval/launch_eval_runs_meissonic_ft.py \
      "${COMMON_ARGS[@]}" \
      --num_particles=$n \
      --batch_p=$n \
      --proposal_type="reverse" \
      --phi=$phi \
      --lora_ckpt_uuid="9e7bb4ef-4aa8-43c5-8677-f6636098972c" \
      --reward_name="hps" \
      --guidance_reward_fn="HumanPreference"
  done
done



# SMC
# for phi in 1; do
#   for n in 2 4 8 16; do
#     echo "▶ Running SMC with phi = ${phi}, and n=${n}"
#     python src/eval/launch_eval_runs_meissonic_ft.py \
#       "${COMMON_ARGS[@]}" \
#       --num_particles=$n \
#       --batch_p=$n \
#       --proposal_type="reverse" \
#       --phi=$phi
#   done
# done
