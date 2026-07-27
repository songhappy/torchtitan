#!/bin/bash
#PBS -l select=4:ncpus=208
#PBS -l walltime=00:30:00
#PBS -q workq
#PBS -N grpo_xpu_4n
#PBS -o /home/guoqiong/git/torchtitan/torchtitan/experiments/rl/grpo_xpu_4n.out
#PBS -e /home/guoqiong/git/torchtitan/torchtitan/experiments/rl/grpo_xpu_4n.err

set +e

source ~/env-3.sh
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate monarch

export ZE_AFFINITY_MASK=0,1,2,3
export CCL_OFI_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64/libfabric.so.1
export FI_PROVIDER=cxi
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_OFI_PROVIDER=cxi
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_xpu
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

HF_ASSETS_PATH=${HF_ASSETS_PATH:-/home/guoqiong/models/Qwen3-0.6B}
PPN=${PPN:-4}
REQUIRED_NODES=4

# Clear stale checkpoint and inductor cache (node-local, cleared via mpiexec below)
rm -rf ~/git/torchtitan/outputs/rl/checkpoint/ 2>/dev/null

# Known bad nodes (broken Level Zero driver, 0 XPU devices).
BAD_NODES="x1001c1s1b0n0"

# Filter out bad nodes from allocation.
ALL_NODES=""
NUM_NODES=0
for node in $(sort -u "$PBS_NODEFILE"); do
    short=${node%%.*}
    if echo "$BAD_NODES" | grep -qw "$short"; then
        echo "Excluding known bad node: $node"
        continue
    fi
    if [ -z "$ALL_NODES" ]; then
        ALL_NODES="$node"
    else
        ALL_NODES="${ALL_NODES},${node}"
    fi
    NUM_NODES=$((NUM_NODES + 1))
done

echo "=== Multi-node GRPO: ${NUM_NODES} healthy nodes (need ${REQUIRED_NODES}) ==="
echo "Nodes: ${ALL_NODES}"

if [ "$NUM_NODES" -lt "$REQUIRED_NODES" ]; then
    echo "ERROR: Only ${NUM_NODES} healthy nodes, need ${REQUIRED_NODES}. Aborting."
    exit 1
fi

# Take exactly REQUIRED_NODES nodes.
ALL_NODES=$(echo "$ALL_NODES" | tr ',' '\n' | head -n "$REQUIRED_NODES" | tr '\n' ',' | sed 's/,$//')
NUM_NODES=$REQUIRED_NODES

echo "Using ${NUM_NODES} nodes: ${ALL_NODES}"

cd ~/git/torchtitan

# Clear inductor cache on all compute nodes for a fresh run
mpiexec -n "$NUM_NODES" -ppn 1 --hosts "$ALL_NODES" \
    bash -c "rm -rf /tmp/torchinductor_xpu"

mpiexec -n "$NUM_NODES" -ppn 1 --hosts "$ALL_NODES" --envall \
    python3 -m torchtitan.experiments.rl.multinode_launcher \
    --num_nodes="$NUM_NODES" \
    --gpus_per_node="$PPN" \
    --all_nodes="$ALL_NODES" \
    --module alphabet_sort --config rl_grpo_lora_qwen3_0_6b \
    --hf_assets_path="$HF_ASSETS_PATH" \
    > torchtitan/experiments/rl/train_xpu_4n.log 2>&1

echo "Exit code: $?"
