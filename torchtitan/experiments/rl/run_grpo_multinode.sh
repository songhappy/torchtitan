#!/bin/bash
#PBS -l select=8
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -q debug-scaling
#PBS -N grpo_full_mn
#PBS -l filesystems=flare:home
#PBS -A Intel-Aurora
#PBS -j oe
#
# Multi-node full-parameter GRPO on Intel XPU (Aurora / PBS). Default: 8 nodes.
#
# Submit:
#   qsub torchtitan/experiments/rl/run_grpo_multinode.sh
# Fewer/more nodes (the -l select on the command line wins over the header):
#   qsub -l select=4 torchtitan/experiments/rl/run_grpo_multinode.sh
# Interactive:
#   qsub -I -l select=8 -l walltime=01:00:00 -A <account> -q debug-scaling
#   bash torchtitan/experiments/rl/run_grpo_multinode.sh
#
# Overridable via env: NUM_NODES (use only the first N of the allocation), PPN
# (GPUs per node), TP, DP_REPLICATE, CONFIG, NUM_STEPS, VAL_SAMPLES,
# HF_ASSETS_PATH, DUMP_FOLDER, INTERPOSER, MPIEXEC, EXTRA_ARGS. `qsub -v` is
# needed to forward them into a batch job. Extra args go to the training command
# verbatim -- but only when run with bash: qsub accepts no trailing script args,
# so under PBS use EXTRA_ARGS instead.
#   TP=2 NUM_STEPS=20 qsub -l select=4 -v TP,NUM_STEPS run_grpo_multinode.sh
#   NUM_NODES=2 bash run_grpo_multinode.sh --async_loop.group_size=16
#   EXTRA_ARGS=--trainer.checkpoint.load-only qsub -v EXTRA_ARGS ...
set +e

source ~/env-3.sh
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate monarch

export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0,1,2,3}
export CCL_OFI_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64/libfabric.so.1
export FI_PROVIDER=cxi
export CCL_ATL_TRANSPORT=ofi
export CCL_ATL_OFI_PROVIDER=cxi

export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:${LD_LIBRARY_PATH}
export FI_PROVIDER_PATH=/opt/cray/libfabric/1.22.0/lib64/libfabric

# CXI/oneCCL scale-out tuning. Required for correct AND working cross-node runs:
# without it, tp=8 dies on `atl_ofi.cpp:1071 fi_cq_readerr err 5` and tp4/rep2
# trains with a corrupted grad_norm (88-2464 vs a healthy 0.05-0.15).
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_OVFLOW_BUF_SIZE=8388608
export FI_CXI_CQ_FILL_PERCENT=20
export CCL_ALLREDUCE_SCALEOUT=direct
export CCL_BCAST=double_tree
export CCL_SYCL_SCALEOUT_HOST_BUF_SIZE=$((2 * 1024 * 1024 * 1024))
export CCL_OP_SYNC=1
# Progress threads off the compute cores. Longer than the rank count on purpose;
# oneCCL takes what it needs. NOT CCL_WORKER_COUNT, which measured 29% SLOWER.
export CCL_WORKER_AFFINITY="5,13,21,29,37,45,57,65,73,81,89,97"
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export TORCH_LLM_ALLREDUCE=1

# Cray PALS by ABSOLUTE PATH: env-3.sh puts Intel MPI's Hydra mpiexec ahead of
# PALS on PATH, and under Hydra a cross-node tp=8 run dies on a oneCCL SEND fault.
MPIEXEC=${MPIEXEC:-/opt/cray/pals/1.8/bin/mpiexec}

# PALS forwards the head node's env verbatim (--envall), so these two must be
# fixed up or the other nodes inherit values that are wrong for them:
#   TMPDIR - PBS sets /var/tmp/pbs.<jobid>, which exists only on the head node;
#            Monarch's worker bootstrap dies on the missing directory.
#   HOSTNAME - torchstore reads it for shared-memory locality (get_local_hostname),
#            so every rank thinks it is local to the storage volume and a
#            cross-node pull dies on "Shared memory storage not found".
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

# REQUIRED FOR FULL GRPO, not for LoRA. Full GRPO trains the qk_norm weights, so
# its backward asks aten::_fused_rms_norm_backward for a weight gradient; above
# xe_core_count * 1024 rows the stock XPU kernel dies with "tensor does not have
# a device". The interposer supplies the patched kernel without touching the
# conda env. Build it with build_rmsnorm_interposer.sh.
# Absolute, not relative to this file: PBS copies the submitted script into a
# spool dir, so BASH_SOURCE does not resolve back into the repo. Matches the
# ~/git/torchtitan assumption the cd below already makes.
INTERPOSER=${INTERPOSER:-$HOME/git/torchtitan/torchtitan/experiments/rl/rmsnorm_interposer/libinterpose_layernorm.so}
if [ ! -f "$INTERPOSER" ]; then
    echo "FATAL: interposer not found at $INTERPOSER"
    echo "       Run: bash torchtitan/experiments/rl/build_rmsnorm_interposer.sh"
    exit 1
fi

HF_ASSETS_PATH=${HF_ASSETS_PATH:-/flare/Aurora_deployment/intel/models/Qwen3-0.6B}
PPN=${PPN:-4}
NUM_STEPS=${NUM_STEPS:-10}
VAL_SAMPLES=${VAL_SAMPLES:-0}
CONFIG=${CONFIG:-rl_grpo_full_qwen3_0_6b_flex}
DUMP_FOLDER=${DUMP_FOLDER:-outputs/rl_full_multinode}
# Nodes split half trainer / half generator by multinode_launcher. Trainer
# dp_shard = trainer_gpus/(TP*DP_REPLICATE), capped at 4 without LoRA, with the
# excess spilling onto dp_replicate. Keep TP inside a node: cross-node TP is
# correct but ~10x slower.
TP=${TP:-1}
DP_REPLICATE=${DP_REPLICATE:-1}
# qsub takes no trailing script arguments (unlike sbatch), so "$@" is always
# empty under PBS. EXTRA_ARGS is the only way to reach the launcher's flags from
# a qsub -v, e.g. EXTRA_ARGS=--trainer.checkpoint.load-only to skip the DCP save
# that OOMs at dp_shard=16. Use load-only, NOT no-enable: `enable` also gates the
# initial HF weight load, so no-enable silently trains from random init and the
# first step never completes. Word-split on purpose so several flags can be passed.
EXTRA_ARGS=${EXTRA_ARGS:-}
# The live log name defaults to per-node-count, which COLLIDES when two full-param
# runs of the same size overlap (e.g. a 10-step smoke test and a 200-step run):
# the second truncates the first, and the watcher archives whichever it happens to
# read. Override it per run to keep them apart.
LOG=${LOG:-torchtitan/experiments/rl/train_full_${NUM_NODES}n.log}

# A stale checkpoint from a different mesh shape fails to load, so start clean.
rm -rf ~/git/torchtitan/"$DUMP_FOLDER"/checkpoint/ 2>/dev/null
rm -rf ~/.cache/torchinductor_xpu/triton 2>/dev/null

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

echo "=== Multi-node full GRPO: ${NUM_NODES}/${AVAILABLE_NODES} nodes, ${PPN} GPUs/node ==="
echo "Nodes:      ${ALL_NODES}"
echo "Config:     ${CONFIG}  TP=${TP}  DP_REPLICATE=${DP_REPLICATE}"
echo "LD_PRELOAD: ${INTERPOSER}"

cd ~/git/torchtitan || { echo "ERROR: ~/git/torchtitan not found"; exit 1; }

# Confirm on this node that the interposer takes effect and the previously fatal
# shape survives, before spending the allocation. A silently ineffective preload
# would only surface as the original crash partway into the run.
env "LD_PRELOAD=$INTERPOSER" python3 -c "
import torch
assert 'libinterpose_layernorm.so' in open('/proc/self/maps').read(), 'LD_PRELOAD did not take effect'
x = torch.randn(2, 2048, 16, 128, device='xpu', dtype=torch.bfloat16, requires_grad=True)
torch.nn.RMSNorm(128, eps=1e-6, dtype=torch.bfloat16).to('xpu')(x).sum().backward()
torch.xpu.synchronize()
print('interposer active; 65536-row RMSNorm weight-grad backward: OK')
"
if [ $? -ne 0 ]; then
    echo "FATAL: interposer precheck failed, not starting the pipeline."
    exit 1
fi

# --cpu-bind none is REQUIRED. PALS binds each rank to its own core slice, and at
# -ppn 1 that slice is a SINGLE core; every Monarch actor and vLLM worker forked
# from the launcher inherits the mask, so ~126 threads share 1 core of 204. Costs
# 2-4x end-to-end (generator ITL 55 ms -> 215-260 ms).
#
# LD_PRELOAD is set on the launched command via `env` rather than exported here,
# so the interposer lands in each node's python process (and everything it forks)
# without also being preloaded into mpiexec itself.
"$MPIEXEC" -n "$NUM_NODES" -ppn 1 --hosts "$ALL_NODES" --cpu-bind none --envall \
    env "LD_PRELOAD=$INTERPOSER" \
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
echo "=== rmsnorm crashes (must be 0) ==="
grep -c "tensor does not have a device" "$LOG"
echo "=== steps completed ==="
grep -E "Train \| Step" "$LOG" | tail -5
