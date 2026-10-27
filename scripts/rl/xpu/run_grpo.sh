#!/bin/bash
#PBS -l select=2
#PBS -l place=scatter
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -N grpo
#PBS -l filesystems=flare:home
#PBS -A Intel-Aurora
#PBS -j oe
#
# GRPO on Aurora (Intel XPU, PBS) with torchtitan.rl.train on any node count, through
# torchtitan.rl.actors.monarch.job_launcher:
#   qsub -l select=8 -v ARM=lora,PPN=4,NUM_STEPS=10 scripts/rl/xpu/run_grpo.sh
# Runs rl_grpo_qwen3_0_6b_<ARM>_<nodes>n_ppn<PPN>_<NUM_STEPS>steps from
# torchtitan_recipes/rl/alphabet_sort.py unless CONFIG is set. The model is read
# from assets/hf/Qwen3-0.6B (a link to the shared copy).
# TRN_COPIES=<R> runs the _trn_c<R> variant (R copies, FSDP across nodes), e.g. 8 nodes:
#   qsub -l select=8 -v ARM=full,PPN=4,NUM_STEPS=2,TRN_COPIES=1 scripts/rl/xpu/run_grpo.sh
# GEN_TP=<T> spreads each vLLM generator over T tiles (the _gen_tp<T> variant).
# Test one side, the other at its default on one full node (2 nodes x 12): TRN_TILES=<A> (A trainer tiles,
# _trn_t<A>), GEN_TILES=<E> (E generator tiles, _gen_t<E>). BATCH_SCALED=1 sizes the batch to the trainer.
# UPSTREAM=1 (1 node) runs the base recipe's layout: trainer TP 2, one generator TP 2 (_upstream).
# Overridable: ARM PPN NUM_STEPS TRN_COPIES GEN_TP TRN_TILES GEN_TILES BATCH_SCALED UPSTREAM CONFIG DUMP_FOLDER LOG MPIEXEC EXTRA_ARGS.

# The checkout qsub ran from (cd there first), so a second clone runs its own code.
TORCHTITAN_DIR=${TORCHTITAN_DIR:-${PBS_O_WORKDIR:-$HOME/git/torchtitan}}
source "${ONEAPI_ENV_SCRIPT:-$TORCHTITAN_DIR/scripts/rl/xpu/env_torch214.sh}"
eval "$("${CONDA_PREFIX_BASE:-$HOME/miniforge3}/bin/conda" shell.bash hook)"
conda activate "${CONDA_ENV:-monarch_t215}"  # torch 2.15: upstream main needs it

ARM=${ARM:-full}
PPN=${PPN:-4}
NUM_STEPS=${NUM_STEPS:-10}
MPIEXEC=${MPIEXEC:-/opt/cray/pals/1.8/bin/mpiexec}  # PALS, not Intel MPI's Hydra
EXTRA_ARGS=${EXTRA_ARGS:-}  # qsub passes no script args; extra flags go through here
mapfile -t NODES < <(sed 's/\..*//' "${PBS_NODEFILE:?run inside a PBS job}" | awk '!seen[$0]++')
NUM_NODES=${#NODES[@]}
CONFIG=${CONFIG:-rl_grpo_qwen3_0_6b_${ARM}_${NUM_NODES}n_ppn${PPN}${TRN_COPIES:+_trn_c$TRN_COPIES}${GEN_TP:+_gen_tp$GEN_TP}${TRN_TILES:+_trn_t$TRN_TILES}${GEN_TILES:+_gen_t$GEN_TILES}${BATCH_SCALED:+_batch_scaled}${UPSTREAM:+_upstream}_${NUM_STEPS}steps}
DUMP_FOLDER=${DUMP_FOLDER:-outputs/${CONFIG}}
LOG=${LOG:-$DUMP_FOLDER/train.log}

export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-$(seq -s, 0 $((PPN - 1)))}
LIBFABRIC_ROOT=$(ls -d /opt/cray/libfabric/*/ 2>/dev/null | sort -V | tail -1)
LIBFABRIC_ROOT=${LIBFABRIC_ROOT%/}  # newest: system refreshes rotate the version dir
[ -e "$LIBFABRIC_ROOT/lib64/libfabric.so.1" ] || { echo "ERROR: no Cray libfabric"; exit 1; }
export CCL_OFI_LIBRARY_PATH=$LIBFABRIC_ROOT/lib64/libfabric.so.1
export LD_LIBRARY_PATH=$LIBFABRIC_ROOT/lib64:$LD_LIBRARY_PATH
export FI_PROVIDER_PATH=$LIBFABRIC_ROOT/lib64/libfabric
export FI_PROVIDER=cxi CCL_ATL_TRANSPORT=ofi CCL_ATL_OFI_PROVIDER=cxi
# CXI queues and oneCCL scale-out algorithms that cross-node runs need.
export FI_CXI_DEFAULT_CQ_SIZE=131072 FI_CXI_OVFLOW_BUF_SIZE=8388608 FI_CXI_CQ_FILL_PERCENT=20
export CCL_ALLREDUCE_SCALEOUT=direct CCL_BCAST=double_tree
# Per-communicator host buffer; the weight pull keeps PPN^2 communicators per node.
export CCL_SYCL_SCALEOUT_HOST_BUF_SIZE=${CCL_SYCL_SCALEOUT_HOST_BUF_SIZE:-$((256 * 1024 * 1024))}
unset CCL_OP_SYNC CCL_WORKER_AFFINITY  # oneCCL#223; explicit affinity aborts oneCCL 2022
# oneCCL#222: a cross-node broadcast moves no data when the process's first communicator was
# intra-node (e.g. a TP generator's); the scheduler path, taken at threshold 0, is correct.
export CCL_SYCL_BROADCAST_SCALEOUT_THRESHOLD=0
export SYCL_UR_USE_LEVEL_ZERO_V2=1  # the v1 adapter aborts intermittently (ur_die)
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1 TORCH_LLM_ALLREDUCE=1
export TMPDIR=/tmp  # PBS's TMPDIR exists only on the head node
unset HOSTNAME      # else every node looks local to TorchStore
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-hsn0}
# Inductor and Triton cache under $TMPDIR (node-local) by default; a cache on /home shared by
# every node races: a rank reads back a kernel file another node is replacing and fails to compile.
export VLLM_ENABLE_V1_MULTIPROCESSING=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled
# Spawning a host's procs (torch + vLLM imports) takes longer than Monarch's 30 s default.
export HYPERACTOR_HOST_SPAWN_READY_TIMEOUT=300s HYPERACTOR_MESH_PROC_SPAWN_MAX_IDLE=300s

cd "$TORCHTITAN_DIR" || exit 1
mkdir -p "$DUMP_FOLDER"
rm -rf "$DUMP_FOLDER/checkpoint"  # a checkpoint from another mesh shape fails to load
echo "=== $CONFIG on ${NODES[*]} (tiles $ZE_AFFINITY_MASK), log $LOG ==="

# The client finds its hosts through this job, as with SlurmJob's batch mode.
export TORCHTITAN_MONARCH_JOB=$DUMP_FOLDER/monarch_job.pkl
python3 -c 'import sys; from monarch.job import PBSJob; PBSJob({"hosts": int(sys.argv[1])}).dump(sys.argv[2])' \
    "$NUM_NODES" "$TORCHTITAN_MONARCH_JOB" || exit 1
# Every RL teardown aborts and would dump ~10 GB of core per process; PALS exports the limit.
ulimit -c 0
# One Monarch worker per node, then the client here. --cpu-bind none: PALS would pin each
# worker and every proc it forks to one core.
python3 -m monarch._src.job._pbs_batch --mpiexec "$MPIEXEC" --rank-env PALS_RANKID \
    --mpiexec-arg=--cpu-bind=none \
    "python3 -m torchtitan.rl.actors.monarch.job_launcher --module torchtitan_recipes.rl.alphabet_sort \
    --config $CONFIG --output-dir $DUMP_FOLDER $EXTRA_ARGS $*" > "$LOG" 2>&1
STATUS=$?
grep -E "Train \| Step" "$LOG" | tail -3
exit $STATUS
