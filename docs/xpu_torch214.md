# GRPO on Intel XPU (Aurora, torch 2.14)

This branch runs end-to-end GRPO on Aurora (Intel Data Center GPU Max 1550) with the
upstream torchtitan RL stack -- Monarch actors, TorchStore weight sync, vLLM generation,
FSDP2 training -- on **torch 2.14.0+xpu and oneCCL 2022**. It is written for a team that
wants to rebuild the stack, run the reference workloads and check that a run is healthy.

The XPU support follows the CUDA code paths; differences are decided where they matter
(device type checks in the generator, worker and launch env, oneCCL workarounds that
default on for XPU). The only Aurora-specific pieces are the launcher
(`scripts/rl/xpu/run_grpo.sh`, which uses Monarch's `PBSJob` the way `SlurmJob` is used on
Slurm), the environment script (`scripts/rl/xpu/env_torch214.sh`) and the layout recipes.

Contents: [Quick start](#quick-start) - [Configurations](#configurations) -
[Running](#running) - [A healthy run](#a-healthy-run) - [Reference results](#reference-results) -
[Environment setup](#environment-setup) - [XPU settings and why](#xpu-settings-and-why) -
[Known issues](#known-issues)

## Quick start

With the environment built ([Environment setup](#environment-setup)), from this checkout:

```bash
export HF_ASSETS_PATH=/flare/Aurora_deployment/intel/models/Qwen3-0.6B
mkdir -p torchtitan/rl/example_checkpoint   # the recipes read the model from here
ln -sfn $HF_ASSETS_PATH torchtitan/rl/example_checkpoint/Qwen3-0.6B
qsub -l select=1 -v ARM=lora scripts/rl/xpu/run_grpo.sh      # 1 node, 10 steps
tail -f outputs/rl_grpo_qwen3_0_6b_lora_1n_ppn4_10steps/train.log
```

Expect ten `Train | Step: N` lines; the job then ends by itself, with exit status 1
([teardown](#known-issues)).

## Configurations

All runs use `rl_grpo_qwen3_0_6b_flex` from `torchtitan_recipes/rl/alphabet_sort.py`
(Qwen3-0.6B, alphabet-sort task, 0/1 reward) in two arms that differ only in the
`LoRATransform`:

| Arm | Config | Trains |
|-----|--------|--------|
| Full GRPO | `rl_grpo_qwen3_0_6b_full_<N>n_ppn<P>_<S>steps` | every parameter |
| LoRA GRPO | `rl_grpo_qwen3_0_6b_lora_<N>n_ppn<P>_<S>steps` | rank-32 adapters on `wqkv`, `wo` |

`N` is the node count (1, 2, 4, 8, 16), `P` the tiles per node (1, 2, 4, 6, 8, 10, 12;
default 4) and `S` the steps (2, 10, 200). The recipe fixes the layout:

- **1 node**: the trainer and the generators share the node's tiles (2 + 2 at `P=4`).
- **N nodes**: the trainer takes the first N/2 nodes, single-tile vLLM generators the rest.
  Each trainer node holds one model copy: `dp_replicate` = trainer nodes, `dp_shard` =
  tiles per node, so FSDP's per-layer all-gather stays inside a node. At `P=10`, where
  `dp_shard=10` cannot shard the FFN weight (3072), the recipe picks the smallest
  `dp_replicate` that can.

For a model too large for one node, FSDP has to shard one copy across nodes. Every layout
on 4 or more nodes also has `_rep1` and `_rep2` variants (select them with
`DP_REPLICATE=<R>`): R copies, each sharded across `trainer GPUs / R` tiles. For example:

| Config | Nodes | dp_replicate x dp_shard | Nodes per model copy |
|--------|-------|-------------------------|----------------------|
| `rl_grpo_qwen3_0_6b_{full,lora}_16n_ppn4_rep2_2steps` | 16 | 2 x 16 | 4 |
| `rl_grpo_qwen3_0_6b_{full,lora}_16n_ppn4_rep1_2steps` | 16 | 1 x 32 | 8 |
| `rl_grpo_qwen3_0_6b_{full,lora}_8n_ppn4_rep1_2steps` | 8 | 1 x 16 | 4 |
| `rl_grpo_qwen3_0_6b_{full,lora}_8n_ppn4_rep2_2steps` | 8 | 2 x 8 | 2 (full arm hangs, see [Known issues](#known-issues)) |

`_gtp<T>` variants (select with `GEN_TP=<T>`) spread each vLLM generator over T tiles, e.g.
`rl_grpo_qwen3_0_6b_full_2n_ppn4_gtp2_2steps`. Across nodes they need a oneCCL setting (see
[Known issues](#known-issues)).

## Running

`scripts/rl/xpu/run_grpo.sh` is one PBS script for every node count. Submit it from the
checkout -- it runs the code of the directory `qsub` was run from -- and pass overrides
with `qsub -v`:

```bash
qsub -l select=1 -v ARM=lora scripts/rl/xpu/run_grpo.sh
qsub -l select=8 -q debug-scaling -v ARM=full,PPN=4 scripts/rl/xpu/run_grpo.sh
qsub -l select=8 -l walltime=04:00:00 -q capacity -v NUM_STEPS=200 scripts/rl/xpu/run_grpo.sh
qsub -l select=8 -q capacity -v ARM=full,PPN=4,NUM_STEPS=2,DP_REPLICATE=1 scripts/rl/xpu/run_grpo.sh
```

| Override | Default | Meaning |
|----------|---------|---------|
| `ARM` | `full` | `full` or `lora` |
| `PPN` | `4` | tiles per node; 1 or an even number up to 12 |
| `NUM_STEPS` | `10` | 2, 10 or 200 |
| `DP_REPLICATE` | unset | select the `_rep<R>` variant (FSDP across nodes) |
| `CONFIG` | from the above | any config in `alphabet_sort.py` |
| `DUMP_FOLDER`, `LOG` | `outputs/<config>`, `$DUMP_FOLDER/train.log` | output locations |
| `CONDA_ENV` | `monarch_pbs` | the env from [Environment setup](#environment-setup) |
| `TORCHTITAN_DIR`, `ONEAPI_ENV_SCRIPT`, `MPIEXEC`, `EXTRA_ARGS` | see the script | rarely needed |

Per-user queue limits on Aurora: `debug` (<= 2 nodes) and `debug-scaling` take one
running and one queued job each, `capacity` (<= 16 nodes) two running and five in total.
A `qsub` past that is refused with `would exceed queue generic's per-user limit`.

How it launches: the script dumps a `monarch.job.PBSJob({"hosts": <nodes>})` and points
`TORCHTITAN_MONARCH_JOB` at it; `python -m monarch._src.job._pbs_batch` starts one Monarch
worker per node with Cray PALS `mpiexec` (`SlurmJob` uses `srun`) and runs
`torchtitan.rl.train` as the client, which places the trainer and generators on the job's
hosts.

## A healthy run

- every requested step logs a `Train | Step: N` line;
- reward is non-zero from step 1. An occasional `Consecutive untrainable batches: 1/10` is
  normal (every sample of a group got the same reward); reward 0 with that count climbing
  toward 10 means the generators received zero weights;
- `grad_norm` in the ranges below: LoRA about 0.02-0.5, full about 0.2-4 (full is taken
  over all parameters; compare it only with full runs);
- the `Mesh split` line in `train.log` shows the mesh you expect -- read the mesh from that
  line, not from the requested flags.

## Reference results

**Regression, 2026-10-14** (`grpo_regression` skill, `sweep.sh basic`, on torchtitan `5ae08f010`, Monarch
`xpu-all` `5009e95d5`, TorchStore `xpu-all` `2afb113`): 18/18 pass --

- 2 nodes x 1, 2, 4, 8, 12 tiles, full and LoRA (the whole trainer on one node);
- 16 nodes x 4 and 12 tiles, full and LoRA, one model copy per trainer node (`8 x 4`, `8 x 12`);
- the four 10-step runs below.

FSDP across nodes, same day: `1 x 8` (4 nodes) and `1 x 16` (8 nodes) pass in both arms, `2 x 8`
(8 nodes) passes with LoRA and hangs with full GRPO (see [Known issues](#known-issues)).

10-step runs, 4 tiles per node (throughput is the median of steps 3-10):

| Run | Nodes | Mesh | Steps | `fwd_bwd` tok/s | full step tok/s | grad_norm | mean reward |
|-----|-------|------|-------|-----------------|-----------------|-----------|-------------|
| Full | 1 | 1 x 2 | 10/10 | 2712 | 2412 | 0.16-2.92 | 0.29 |
| LoRA | 1 | 1 x 2 | 10/10 | 3038 | 2714 | 0.05-0.38 | 0.28 |
| Full | 8 | 4 x 4 | 10/10 | 2322 | 2097 | 0.26-2.23 | 0.30 |
| LoRA | 8 | 4 x 4 | 10/10 | 5895 | 1469 | 0.04-0.28 | 0.28 |

200-step runs, 2026-10-15, same code and layouts (throughput is the median of steps 3-200; reward is
the mean of the first / last 10 steps):

| Run | Nodes | Mesh | Steps | `fwd_bwd` tok/s | full step tok/s | grad_norm | mean reward |
|-----|-------|------|-------|-----------------|-----------------|-----------|-------------|
| Full | 1 | 1 x 2 | 200/200 | 2801 | 2370 | 0.08-2.02 | 0.29 -> 0.41 |
| LoRA | 1 | 1 x 2 | 200/200 | 3495 | 2180 | 0.01-0.40 | 0.26 -> 0.39 |
| Full | 8 | 4 x 4 | 200/200 | 2302 | 1702 | 0.06-1.94 | 0.28 -> 0.41 |
| LoRA | 8 | 4 x 4 | 200/200 | 5976 | 1655 | 0.007-0.29 | 0.26 -> 0.42 |

Earlier 200-step runs on 8 nodes checksummed every TorchStore weight transfer on both sides
(a study-only hook): all 12819 (full) and 51213 (LoRA) transfers were byte-exact.

## Environment setup

Build on the UAN (login node), never on a compute node. Prerequisites: Aurora access,
miniforge3, the Qwen3-0.6B weights on a filesystem the compute nodes read, and the
dataset cached once on the UAN (runs set `HF_HUB_OFFLINE=1`):
`python -c "import datasets; datasets.load_dataset('kalomaze/alphabetic-arxiv-authors-it1', split='train')"`.

```bash
export TORCHTITAN_DIR=$HOME/git/torchtitan    # this branch
export MONARCH_DIR=$HOME/git/monarch
export TORCHSTORE_DIR=$HOME/git/torchstore
export VLLM_DIR=$HOME/git/vllm
export CONDA_PREFIX_BASE=$HOME/miniforge3
```

| Repo | Branch | Commit | Contents |
|------|--------|--------|----------|
| torchtitan | this branch | -- | this guide |
| monarch | `xpu-all` (github.com/songhappy/monarch) | `5009e95d5` | upstream `main` `5283c8570` + PR #4307 (XPU actor runtime) + the `monarch.job.PBSJob` PR (`pbs-job`) |
| torchstore | `xpu-all` (github.com/songhappy/torchstore) | `2afb113` | PR #171 (XCCL transport) + PR #201 (replicated regions fetched once, spread over replicas) |
| vLLM | upstream `main` | `083060d04` | no local patches |

### 1. Conda env and toolchain

```bash
source $TORCHTITAN_DIR/scripts/rl/xpu/env_torch214.sh   # oneAPI 2026.1, gcc 14.3
$CONDA_PREFIX_BASE/bin/conda create -n monarch_pbs python=3.12 -y   # the launcher's CONDA_ENV
eval "$($CONDA_PREFIX_BASE/bin/conda shell.bash hook)" && conda activate monarch_pbs
conda install -y -c conda-forge rust=1.97.1
```

Never source a oneAPI 2025.x environment in the same shell: torch 2.14 ships the 2026.1
SYCL runtime, and two SYCL runtimes in one process segfault.

### 2. torch, Triton and the vLLM XPU kernels

vLLM `main` pins the matching set in `requirements/xpu.txt` (`torch==2.14.0`,
`triton==3.8.0+xpu`, `vllm_xpu_kernels==0.1.15.4`):

```bash
git clone https://github.com/vllm-project/vllm.git $VLLM_DIR
cd $VLLM_DIR && git checkout 083060d04
pip install -r requirements/xpu.txt \
    --extra-index-url https://download.pytorch.org/whl/xpu --extra-index-url https://wheels.vllm.ai/xpu/
python -c "import torch; print(torch.__version__)"   # 2.14.0+xpu (xpu.is_available() is False on a UAN)
```

If `import triton.language` later fails (`module 'triton' has no attribute 'language'`),
another package's `triton` overwrote `triton-xpu`:
`pip install --force-reinstall --no-deps triton-xpu==3.8.0`.

### 3. vLLM

```bash
cd $VLLM_DIR && VLLM_TARGET_DEVICE=xpu VLLM_REQUIRE_RUST_FRONTEND=1 \
    pip install -e . --no-deps --no-build-isolation -v
```

`VLLM_REQUIRE_RUST_FRONTEND=1` makes a missing `cargo` an error instead of silently
dropping `_rust_tool_parser`.

### 4. Monarch

```bash
git clone https://github.com/songhappy/monarch.git $MONARCH_DIR && cd $MONARCH_DIR
git checkout xpu-all   # upstream main + PR #4307 + PBSJob
GCC=/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/gcc-13.4.0-22p27zg/bin   # the GNU gcc this was built with
LLVM=/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/llvm-develop-git.87717bf-7gy6r7k
RUSTC_BOOTSTRAP=1 USE_TENSOR_ENGINE=0 \
PROTOC=$CONDA_PREFIX/lib/python3.12/site-packages/torch/bin/protoc \
CC=$GCC/gcc CXX=$GCC/g++ CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1" \
CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=$GCC/gcc \
LIBCLANG_PATH=$LLVM/lib BINDGEN_EXTRA_CLANG_ARGS="-I$LLVM/lib/clang/23/include" \
    pip install -e . --no-deps --no-build-isolation
pip install pyzmq pyarrow requests numpy pyre-extensions "typing-extensions>=4.12" \
    cloudpickle lark tabulate opentelemetry-api clusterscope "flask>=2.0" xxhash py-spy aiohttp
python -c "from monarch.actor import Actor; from monarch.job import PBSJob; import monarch._rust_bindings"
```

Why: `USE_TENSOR_ENGINE=0` builds only the actor runtime RL uses; `RUSTC_BOOTSTRAP=1` lets
stable rustc take the nightly flags `setup.py` passes; `PROTOC` is the one torch bundles;
`CC`/`CXX` must be GNU gcc -- 13.4 is the tested one (the oneAPI env sets `CC=icx`, whose
intrinsics GNU `ld` cannot link); the cargo linker override replaces conda's `x86_64-conda-linux-gnu-cc`, whose
sysroot lacks the glibc 2.38 symbols `aws-lc-sys` needs; `LIBCLANG_PATH` and
`BINDGEN_EXTRA_CLANG_ARGS` let `libsqlite3-sys` (upstream telemetry) find libclang.

### 5. TorchStore

```bash
git clone -b xpu-all https://github.com/songhappy/torchstore.git $TORCHSTORE_DIR
cd $TORCHSTORE_DIR && git checkout 2afb113
pip install -e . --no-deps --no-build-isolation && pip install pygtrie portpicker
```

`--no-deps` because TorchStore pins an older `torchmonarch`.

### 6. TorchTitan deps

```bash
cd $TORCHTITAN_DIR
pip install torchdata==0.11.0 spmd_types==0.2.5 tyro==1.0.15 einops tensorboard \
    "git+https://github.com/PrimeIntellect-ai/renderers.git@main" --constraint <(echo "torch==2.14.0+xpu")
```

TorchTitan runs from the source tree and is not pip-installed; the constraint keeps pip
from replacing the XPU torch.

### Reference versions

| Package | Version | Package | Version |
|---------|---------|---------|---------|
| python | 3.12.13 | torchmonarch | 0.8.0.dev0 (editable) |
| torch | 2.14.0+xpu | torchstore | 0.0.0.dev0 (editable) |
| triton-xpu | 3.8.0 | transformers | 5.18.0 |
| oneccl | 2022.1.1 | datasets | 4.7.0 |
| intel-sycl-rt | 2026.1.0 | renderers | 0.1.11 |
| vllm | 0.30.1rc1.dev523+g083060d04.xpu (editable) | tyro | 1.0.15 |
| vllm-xpu-kernels | 0.1.15.4 | spmd_types / torchdata | 0.2.5 / 0.11.0 |

## XPU settings and why

**oneCCL workarounds in the code.** Two oneCCL bugs, filed with standalone C reproducers,
are worked around by config fields that default to on when the device is XPU:

1. **Cross-node broadcast moves no data** ([#222](https://github.com/uxlfoundation/oneCCL/issues/222),
   a regression from oneCCL 2021.17.2). oneCCL fixes a process's position on the node at its
   first communicator and routes later cross-node broadcasts through a sub-group built from
   it. A TorchStore storage volume inside an FSDP trainer process inherits the FSDP position,
   so its broadcasts from tiles 1-3 succeed and deliver nothing: zero weights, reward 0.
   `Controller.Config.storage_volumes_in_own_procs` runs the volumes in a twin of the
   trainer mesh (same hosts and tiles) whose first communicator is a weight-pull one.
2. **Concurrent communicator creation fails** ([#225](https://github.com/uxlfoundation/oneCCL/issues/225))
   in topology discovery (`zesFabricPortGetConfig` -> `ZE_RESULT_ERROR_NOT_AVAILABLE`).
   `InterGeneratorRouter.Config.sequential_initial_pull` runs the first weight pull one
   generator at a time (26-160 s once, at 4-8 nodes); later pulls stay concurrent.

A third bug, a SIGSEGV past ~9 communicators per process on the scheduler path
([#224](https://github.com/uxlfoundation/oneCCL/issues/224)), is avoided by the default SYCL
path: do not set `CCL_SYCL_BROADCAST_SCALEOUT_THRESHOLD=0` or `CCL_ENABLE_SYCL_KERNELS=0`.

**Environment the launcher sets, and why:**

| Setting | Why |
|---------|-----|
| `CCL_OP_SYNC` unset | on oneCCL 2022 it turns cross-node `all_reduce` above 4 MiB into a silent no-op ([#223](https://github.com/uxlfoundation/oneCCL/issues/223)): HSDP replicas then train apart while the metrics look normal |
| `CCL_WORKER_AFFINITY` unset | oneCCL 2022 aborts communicator init on an explicit list |
| `CCL_SYCL_SCALEOUT_HOST_BUF_SIZE=256 MiB` | each weight-pull communicator holds this host buffer; the default 2 GiB fills host memory at 12 tiles per node |
| `SYCL_UR_USE_LEVEL_ZERO_V2=1` | the v1 Level Zero adapter aborts about once per 100-150 steps (`urEventWait must not be called for an internal event`); not root-caused |
| `ZE_FLAT_DEVICE_HIERARCHY=FLAT` (`env_torch214.sh`) | one device per tile; the oneAPI 2026.1 module sets COMPOSITE |
| no `TORCHINDUCTOR_CACHE_DIR` | the compile cache stays under `$TMPDIR=/tmp`, per node; a cache on `/home` shared by all nodes races (see [Known issues](#known-issues)) |
| `TMPDIR=/tmp` | PBS's `TMPDIR` exists only on the head node |
| `unset HOSTNAME` | otherwise every node looks local to TorchStore |
| `mpiexec --cpu-bind=none` | PALS otherwise pins each worker and every process it forks to one core |
| `ulimit -c 0` | every teardown aborts and would write ~10 GB of core per process |

Compared with the earlier torch 2.12 stack: oneAPI 2026.1 instead of 2025.3, oneCCL 2022
instead of 2021.17, vLLM upstream `main` instead of a pinned commit, and the RMSNorm
`LD_PRELOAD` interposer is no longer needed (the stock kernel passes the 65536-row
weight-grad backward).

## Known issues

- **A finished run exits with status 1.** Teardown aborts after the last `Train | Step` line
  (verified 2026-10-14 at 1 and 8 nodes: the job ends by itself within a minute). Judge a run
  by its step lines, not by the PBS exit status.
- **Cross-node FSDP with 2 copies hangs.** `dp_replicate=2` with each copy spanning nodes stops in
  the first `forward_backward`, no error, until walltime: 8n `2 x 8` (full; LoRA passed) and 16n
  `2 x 16` (both arms), seen 2026-10-14. One copy across all trainer nodes (`1 x 16`, `1 x 32`)
  passes. All trainer ranks wait in FSDP's backward all-gather (TorchStore is idle). It reproduces in
  plain PyTorch FSDP2 (no RL) while raw collectives on the same mesh pass, so it is in FSDP2's HSDP
  path on XPU. Use `dp_replicate=1` or one copy per node.
- **One copy over 96 tiles (`1 x 96`, 16n x 12) is at the edge.** Each generator opens one
  communicator per volume shard: host memory reached 966 GB (88% of a node) with LoRA, and full
  GRPO failed its first pull in TorchStore's handshake (`onecclCommInitRank failed with code 11`),
  seen 2026-10-15. `1 x 32` and one copy per node are fine.
- **The first weight pull grows with `dp_shard`.** Each generator opens one communicator per volume
  shard, and the first pull runs one generator at a time (oneCCL #225): at 16n x 12 tiles a
  generator takes ~8 s at `8 x 12`, ~21 s at `2 x 48`, ~40 s at `1 x 96` -- an hour for 96
  generators. Give such runs a long walltime.
- **A hung run with no error in `train.log`** usually means one trainer rank raised and the
  others wait in FSDP collectives. The exception is only in that rank's structured log:
  `grep -l forward_backward_error $DUMP_FOLDER/structured_logs/rl_trainer*`.
- **Never point `TORCHINDUCTOR_CACHE_DIR` at `/home`.** With one cache shared by all nodes,
  at 16 nodes a trainer rank intermittently fails to compile flex attention (`@jit functions
  should be defined in a Python file`) and the run hangs as above.
- **Checkpoint saving is off** (`load_only=True` in the recipe): on the torch 2.12 stack the
  DCP save ran out of device memory inside oneCCL at 8 nodes; not re-tested on 2.14. Do not
  disable the checkpointer instead; that also skips loading the pretrained weights.
- **Generators with TP > 1 across nodes load garbage weights** (oneCCL #222): every completion is
  `!!!!`, every reward 0, and the run stops after 10 untrainable batches (2026-10-15, `_gtp2`/`_gtp4`
  at 2 and 8 nodes). oneCCL fixes a process's tile index at its first communicator, which in a TP
  generator is vLLM's TP group, so the later cross-node weight broadcasts to tiles 1+ move nothing.
  On 1 node (shared-memory pull) `_gtp2` passes. `CCL_SYCL_BROADCAST_SCALEOUT_THRESHOLD=0` makes 2n
  and 8n `_gtp2` pass (set it in the environment you submit from), but it selects the scheduler path of
  #224 (below), so it is not a launcher default until it passes a full regression; the default configs
  use single-tile generators.
- **vLLM with TP > 1 dumps core at process exit** (plain vLLM TP2/TP4, re-seen 2026-10-15) after
  generating correctly; run from a directory with room for the cores, or `ulimit -c 0`.
- Seen with the older Monarch build, not in the 78 runs since the move to Monarch `main`
  + `PBSJob`: a startup flake `cast.service ... actor does not exist`, and a vLLM "not
  enough free memory" abort at startup (a phantom of Level Zero's lagging memory report).
  If either appears, resubmit before changing anything.
