#!/bin/bash
#PBS -l select=8
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -q debug-scaling
#PBS -N grpo_lora_mn
#PBS -l filesystems=flare:home
#PBS -A Intel-Aurora
#PBS -j oe
#
# Multi-node GRPO+LoRA on Intel XPU (Aurora / PBS). Serves any node count: nodes
# come from PBS_NODEFILE and multinode_launcher auto-scales the mesh.
#
# Submit:
#   qsub -l select=4 torchtitan/experiments/rl/run_grpo_lora_multinode.sh
# Interactive:
#   qsub -I -l select=4 -l walltime=01:00:00 -A <account> -q debug-scaling
#   bash torchtitan/experiments/rl/run_grpo_lora_multinode.sh
#
# Overridable via env: NUM_NODES (use only the first N of the allocation), PPN,
# TP, DP_REPLICATE, CONFIG, NUM_STEPS, VAL_SAMPLES, HF_ASSETS_PATH, DUMP_FOLDER,
# MPIEXEC, EXTRA_ARGS. A batch job needs `qsub -v` to forward them. Trailing args
# go to the training command verbatim, but qsub accepts none, so under PBS reach
# the launcher's flags through EXTRA_ARGS instead.
#   NUM_STEPS=20 qsub -l select=4 -v NUM_STEPS run_grpo_lora_multinode.sh
#   NUM_NODES=2 bash run_grpo_lora_multinode.sh --async_loop.group_size=16
#   EXTRA_ARGS=--trainer.checkpoint.load-only qsub -v EXTRA_ARGS ...
set +e

source ~/env-3.sh
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate monarch

# Fabric. Every export below is load-bearing; see GRPO_XPU.md for the bug each
# one closes. env-3.sh points FI_PROVIDER_PATH at a libfabric with no cxi
# provider, so only oneCCL (via CCL_OFI_LIBRARY_PATH) would reach CXI/Slingshot
# and every other consumer would fall back to tcp; force the Cray libfabric for
# all of them. The FI_CXI_*/CCL_* tuning is what makes cross-node runs both
# correct and alive: without it tp=8 dies on `atl_ofi.cpp:1071 fi_cq_readerr
# err 5` and tp4/rep2 trains with grad_norm 88-2464 vs a healthy 0.05-0.15.
export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1,2,3}
export CCL_OFI_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64/libfabric.so.1
export FI_PROVIDER=cxi
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_OFI_PROVIDER=cxi
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:${LD_LIBRARY_PATH}
export FI_PROVIDER_PATH=/opt/cray/libfabric/1.22.0/lib64/libfabric
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_OVFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=20
export CCL_ALLREDUCE_SCALEOUT=direct
export CCL_BCAST=double_tree
export CCL_SYCL_SCALEOUT_HOST_BUF_SIZE=$((2 * 1024 * 1024 * 1024))
export CCL_OP_SYNC=1
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export TORCH_LLM_ALLREDUCE=1
# Progress threads off the compute cores. Longer than the rank count on purpose;
# oneCCL takes what it needs. NOT CCL_WORKER_COUNT, which measured 29% SLOWER.
export CCL_WORKER_AFFINITY="5,13,21,29,37,45,57,65,73,81,89,97"

# Cray PALS by ABSOLUTE PATH: env-3.sh puts Intel MPI's Hydra mpiexec ahead of
# PALS on PATH, and under Hydra a cross-node tp=8 run dies on a oneCCL SEND fault.
MPIEXEC=${MPIEXEC:-/opt/cray/pals/1.8/bin/mpiexec}

# PALS --envall forwards the head node's env verbatim, so anything read from env
# rather than from a syscall must be fixed up first. PBS sets
# TMPDIR=/var/tmp/pbs.<jobid>, which exists only on the head node, and Monarch's
# worker bootstrap dies on the missing directory. torchstore reads HOSTNAME for
# shared-memory locality, so every rank would think it is local to the storage
# volume and a cross-node pull dies on "Shared memory storage not found". Intel
# Hydra masked both by re-running a login shell per node.
export TMPDIR=/tmp
unset HOSTNAME

export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-hsn0}
export TORCHINDUCTOR_CACHE_DIR=~/.cache/torchinductor_xpu
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export MONARCH_ACTOR_QUEUE_DISPATCH=${MONARCH_ACTOR_QUEUE_DISPATCH:-1}

# vLLM's Triton kernels JIT with icpx, which borrows libstdc++ from the gcc on
# PATH; Aurora's default gcc 7.5 lacks <filesystem>. Point icpx at gcc 13.4.
GCC13_ROOT=/opt/aurora/26.26.0/spack/unified/1.1.1/install/linux-x86_64/gcc-13.4.0-hgnyg4p
if [ -d "$GCC13_ROOT" ]; then
    export CCC_OVERRIDE_OPTIONS="+--gcc-toolchain=$GCC13_ROOT"
    export PATH="$GCC13_ROOT/bin:$PATH"
fi

# No LD_PRELOAD interposer here: the patched RMSNorm backward kernel is needed
# only by FULL-parameter GRPO, which trains the qk_norm weights. LoRA freezes
# them, so its backward never asks for that weight gradient. See run_grpo_sn.sh
# / run_grpo_multinode.sh for the full-parameter arm.

HF_ASSETS_PATH=${HF_ASSETS_PATH:-/flare/Aurora_deployment/intel/models/Qwen3-0.6B}
CONFIG=${CONFIG:-rl_grpo_lora_qwen3_0_6b}
NUM_STEPS=${NUM_STEPS:-150}
VAL_SAMPLES=${VAL_SAMPLES:-0}
PPN=${PPN:-4}
DUMP_FOLDER=${DUMP_FOLDER:-outputs/rl_lora_multinode}
# multinode_launcher splits the nodes half trainer / half generator, so
# dp_shard = (NUM_NODES/2 * PPN) / (TP * DP_REPLICATE), capped at the LoRA rank
# with the excess spilling onto dp_replicate. TP=1/DP_REPLICATE=1 (pure dp_shard)
# is the fastest measured shape at every node count tried: at 2 nodes 13-17k
# tok/s vs 8-10k for TP=2 x rep2 and 5.0-5.8k for TP=4. Keep TP inside a node --
# cross-node TP is correct but ~10x slower (tp=8 measured ~500 tok/s).
TP=${TP:-1}
DP_REPLICATE=${DP_REPLICATE:-1}
# qsub takes no trailing script arguments (unlike sbatch), so "$@" is always
# empty under PBS. EXTRA_ARGS is the only way to reach the launcher's flags from
# a qsub -v, e.g. EXTRA_ARGS=--trainer.checkpoint.load-only to skip the DCP save
# that OOMed at dp_shard=16. Use load-only, NOT no-enable: `enable` also gates the
# initial HF weight load, so no-enable silently trains from random init and the
# first step never completes. Word-split on purpose so several flags can be passed.
EXTRA_ARGS=${EXTRA_ARGS:-}

# A stale checkpoint from a different mesh shape fails to load, so start clean.
rm -rf ~/git/torchtitan/"$DUMP_FOLDER"/checkpoint/ 2>/dev/null
rm -rf ~/.cache/torchinductor_xpu/triton 2>/dev/null

cd ~/git/torchtitan || { echo "ERROR: ~/git/torchtitan not found"; exit 1; }
if [ ! -f "$HF_ASSETS_PATH/config.json" ]; then
    echo "ERROR: no model at $HF_ASSETS_PATH (config.json missing)."
    exit 1
fi

# Bare (mgmt) names for both mpiexec --hosts and Monarch --all_nodes. The data
# plane still rides CXI/hsn0 via oneCCL. Do not use .hsn names: they are
# multi-rail and cause MESH_ATTACH_CONFIG_TIMEOUT.
if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE}" ]]; then
    mapfile -t NODE_LIST < <(sort -u "$PBS_NODEFILE" | sed 's/\..*//')
else
    NODE_LIST=("$(hostname -s)")
fi
AVAILABLE_NODES=${#NODE_LIST[@]}
NUM_NODES=${NUM_NODES:-$AVAILABLE_NODES}
if [ "$NUM_NODES" -gt "$AVAILABLE_NODES" ]; then
    echo "ERROR: NUM_NODES=$NUM_NODES but only $AVAILABLE_NODES allocated."
    echo "       Submit with: qsub -l select=$NUM_NODES $0"
    exit 1
fi
ALL_NODES=$(IFS=,; echo "${NODE_LIST[*]:0:$NUM_NODES}")
LOG=torchtitan/experiments/rl/train_lora_${NUM_NODES}n.log

echo "=== GRPO+LoRA: ${NUM_NODES}/${AVAILABLE_NODES} nodes x ${PPN} GPUs, ${CONFIG}," \
     "TP=${TP} DP_REPLICATE=${DP_REPLICATE}, ${NUM_STEPS} steps ==="
echo "Nodes: ${ALL_NODES}"
echo "Log:   ${LOG}"

# --cpu-bind none is REQUIRED. PALS binds each rank to its own core slice, and at
# -ppn 1 that slice is a SINGLE core; every Monarch actor and vLLM worker forked
# from the launcher inherits the mask, so ~126 threads share 1 core of 204. Costs
# 2-4x end-to-end (generator ITL 55 ms -> 215-260 ms). Intel Hydra hid this by
# re-execing a login shell per node, which reset the mask.
"$MPIEXEC" -n "$NUM_NODES" -ppn 1 --hosts "$ALL_NODES" --cpu-bind none --envall \
    python3 -m torchtitan.experiments.rl.multinode_launcher \
    --num_nodes="$NUM_NODES" \
    --gpus_per_node="$PPN" \
    --all_nodes="$ALL_NODES" \
    --module alphabet_sort --config "$CONFIG" \
    --hf_assets_path="$HF_ASSETS_PATH" \
    --async_loop.num_training_steps="$NUM_STEPS" \
    --async_loop.validation.num_samples="$VAL_SAMPLES" \
    --trainer.parallelism.tensor_parallel_degree="$TP" \
    --trainer.parallelism.data_parallel_replicate_degree="$DP_REPLICATE" \
    --dump_folder="$DUMP_FOLDER" \
    $EXTRA_ARGS \
    "$@" \
    > "$LOG" 2>&1

echo "Exit code: $?"
echo "=== steps completed ==="
# Step lines are logged twice, so dedupe; sort NUMERICALLY or "9" outranks "50".
grep -oE 'Train \| Step: *[0-9]+' "$LOG" | grep -oE '[0-9]+' | sort -un | tail -5
