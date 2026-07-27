#!/bin/bash
#PBS -l select=2:ncpus=208
#PBS -l walltime=02:00:00
#PBS -q workq
#PBS -N grpo_xpu_mn
#PBS -o /home/guoqiong/git/torchtitan/torchtitan/experiments/rl/grpo_xpu_2n.out
#PBS -e /home/guoqiong/git/torchtitan/torchtitan/experiments/rl/grpo_xpu_2n.err

set +e

source ~/env-3.sh
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate monarch

export ZE_AFFINITY_MASK=0,1,2,3
export CCL_OFI_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64/libfabric.so.1
export FI_PROVIDER=cxi
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_OFI_PROVIDER=cxi
export TORCHINDUCTOR_CACHE_DIR=~/.cache/torchinductor_xpu
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

HF_ASSETS_PATH=${HF_ASSETS_PATH:-/home/guoqiong/models/Qwen3-0.6B}
PPN=${PPN:-4}

# Clear inductor cache to avoid stale L0 kernel errors on new nodes
rm -rf ~/.cache/torchinductor_xpu/triton 2>/dev/null

NUM_NODES=$(sort -u "$PBS_NODEFILE" | wc -l)
ALL_NODES=$(sort -u "$PBS_NODEFILE" | tr '\n' ',' | sed 's/,$//')

echo "=== Multi-node GRPO: ${NUM_NODES} nodes, ${PPN} GPUs/node ==="
echo "Nodes: ${ALL_NODES}"

cd ~/git/torchtitan

mpiexec -n "$NUM_NODES" -ppn 1 --hosts "$ALL_NODES" --envall \
    python3 -m torchtitan.experiments.rl.multinode_launcher \
    --num_nodes="$NUM_NODES" \
    --gpus_per_node="$PPN" \
    --all_nodes="$ALL_NODES" \
    --module alphabet_sort --config rl_grpo_lora_qwen3_0_6b \
    --hf_assets_path="$HF_ASSETS_PATH" \
    > torchtitan/experiments/rl/train_xpu_2n.log 2>&1

echo "Exit code: $?"
