#!/bin/bash
# XPU 200-step GRPO+LoRA experiment
set -e
source ~/env-3.sh
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate monarch

export ZE_AFFINITY_MASK=0,1,2,3
export FI_PROVIDER=tcp
export CCL_ATL_OFI_PROVIDER=tcp
export TORCHINDUCTOR_CACHE_DIR=~/.cache/torchinductor_xpu
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

HF_ASSETS_PATH=${HF_ASSETS_PATH:-/home/guoqiong/models/Qwen3-0.6B}

cd ~/git/torchtitan

python3 -m torchtitan.experiments.rl.train \
    --module alphabet_sort --config rl_grpo_lora_qwen3_0_6b \
    --hf_assets_path="$HF_ASSETS_PATH" \
    --async_loop.num_training_steps=200 \
    --dump_folder=outputs/rl_lora_200 \
    --generator.gpu_memory_limit=0.90 \
    2>&1 | tee torchtitan/experiments/rl/train_lora_200.log
