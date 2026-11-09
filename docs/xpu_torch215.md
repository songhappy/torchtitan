# GRPO on Intel XPU (Aurora, torch 2.15)

End-to-end GRPO with the upstream torchtitan RL stack (Monarch, TorchStore, vLLM, FSDP2) on Aurora (Intel GPU Max
1550), torch 2.15.0 and oneCCL 2022. This guide rebuilds the stack, runs every reference experiment and gives the
expected result of each. Aurora-specific code: `scripts/rl/xpu/run_grpo.sh` (PBS launcher, Monarch `PBSJob`),
`scripts/rl/xpu/env_torch215.sh` and the layouts in `torchtitan_recipes/rl/alphabet_sort.py`. Gaps vs CUDA:
[xpu_rl_roadmap.md](xpu_rl_roadmap.md).

## Quick start

With the environment built ([Environment setup](#environment-setup)), from this checkout:

```bash
mkdir -p assets/hf   # the recipes read each model from assets/hf/<name>
ln -sfn /flare/Aurora_deployment/intel/models/Qwen3-0.6B assets/hf/Qwen3-0.6B
qsub -l select=1 -v ARM=lora scripts/rl/xpu/run_grpo.sh      # 1 node, 10 steps
tail -f outputs/rl_grpo_qwen3_0_6b_lora_1n_ppn4_10steps/train.log
```

Expect ten `Train | Step: N` lines; the job then ends by itself, with exit status 1 ([teardown](#known-issues)).

## Environment setup

One conda env, `monarch_t215` (the launcher's default). Build on a UAN, not a compute node. Needs: miniforge3, the
model weights where compute nodes can read them, and the dataset cached once (runs are offline):
`python -c "import datasets; datasets.load_dataset('kalomaze/alphabetic-arxiv-authors-it1', split='train')"`.

| Repo | Branch | Commit | Notes |
|---|---|---|---|
| torchtitan | this branch | -- | upstream main + XPU RL commits |
| Monarch | meta-pytorch/monarch `main` | `a0c586486` | no patches (XPU runtime #4307 and `PBSJob` #5067 merged) |
| TorchStore | songhappy/torchstore `xpu-all` | `3edc47f` | upstream `e500a6f` + XCCL transport (#171), replicated fetch (#201), routing fix, stream sync |
| vLLM | vllm-project/vllm `main` | `083060d04` | no patches; newer vLLM dropped the sequence-parallel config torchtitan passes (#60517) |
| vllm-xpu-kernels | `release/0.1.15.4` | -- | built from source for torch 2.15.0 |

### Changes from the previous version of this guide

| Layer | Previous (torch 2.15 nightly) | Now |
|---|---|---|
| torch / torchvision | 2.15.0.dev20261005 / 0.30.0.dev20261005 | **2.15.0 / 0.29.2** (release) |
| Monarch | fork `xpu-all` `2f1637ca6` | **upstream `main` `a0c586486`** |
| triton | triton-xpu 3.9.0 (nightly build) | triton-xpu 3.9.0 + vLLM's `triton==3.9.0+xpu` placeholder |
| vllm-xpu-kernels | 0.1.15.4 from source vs the nightly | same, rebuilt for torch 2.15.0 |
| Build | -- | re-activate the env after step 2 |
| Unchanged | vLLM `083060d04`, TorchStore `3edc47f`, oneAPI 2026.1, oneCCL 2022.1.2 | |

```bash
export TORCHTITAN_DIR=$HOME/git/torchtitan MONARCH_DIR=$HOME/git/monarch TORCHSTORE_DIR=$HOME/git/torchstore
export VLLM_DIR=$HOME/git/vllm KERNELS_DIR=$HOME/git/vllm-xpu-kernels CONDA_PREFIX_BASE=$HOME/miniforge3
```

### 1. Conda env and toolchain

```bash
source $TORCHTITAN_DIR/scripts/rl/xpu/env_torch215.sh   # oneAPI 2026.1, gcc, ZE_FLAT_DEVICE_HIERARCHY=FLAT
$CONDA_PREFIX_BASE/bin/conda create -n monarch_t215 python=3.12 -y
eval "$($CONDA_PREFIX_BASE/bin/conda shell.bash hook)" && conda activate monarch_t215
conda install -y -c conda-forge rust=1.97.1
```

Do not source a oneAPI 2025.x environment in the same shell (two SYCL runtimes segfault).

### 2. torch 2.15.0, Triton and vLLM's other requirements

vLLM's `requirements/xpu.txt` pins torch 2.14; install 2.15.0 and the rest of that file:

```bash
git clone https://github.com/vllm-project/vllm.git $VLLM_DIR && cd $VLLM_DIR && git checkout 083060d04
pip install "torch==2.15.0+xpu" "torchvision==0.29.2+xpu" \
    --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple
pip install "triton==3.9.0+xpu" --no-deps --index-url https://wheels.vllm.ai/xpu/   # placeholder for xgrammar
grep -vE "^(torch|torchaudio|torchvision|triton|vllm_xpu_kernels)\b" requirements/xpu.txt \
    | sed "s#^-r common.txt#-r $VLLM_DIR/requirements/common.txt#" > /tmp/vllm_reqs.txt
pip install -r /tmp/vllm_reqs.txt --constraint <(echo "torch==2.15.0+xpu") \
    --extra-index-url https://download.pytorch.org/whl/xpu
conda deactivate && conda activate monarch_t215   # required, see below
python -c "import torch, triton.language; print(torch.__version__)"   # xpu.is_available() is False on a UAN
```

The re-activation is required: the `oneccl` wheel's activation hooks put the env's `lib/` first on
`LD_LIBRARY_PATH`; without it `import torch` fails (`undefined symbol: urGraphGetIdExp`). The `triton` placeholder
keeps xgrammar from installing CUDA Triton over `triton-xpu`.

### 3. vllm-xpu-kernels from source (~3 h)

Use the GPU compiler (`ocloc`) the kernels' CI pins, not Aurora's: the system one miscompiles flash attention for
PVC with no error (vLLM then generates garbage). The commands unpack the CI's compiler without root:

```bash
git clone -b release/0.1.15.4 https://github.com/vllm-project/vllm-xpu-kernels.git $KERNELS_DIR && cd $KERNELS_DIR
T=$HOME/tools/gpu-ci && mkdir -p $T/debs $T/root $T/bin
python3 -I -c 'import json; c = json.load(open("build_script/gpu_runtime_packages.json"))
print("\n".join(u for u in c["profiles"][c["default"]]["packages"]
      if any(k in u for k in ("intel-igc-core", "intel-igc-opencl", "intel-ocloc_", "libigdgmm"))))' \
    | (cd $T/debs && xargs -n1 curl -sSfLO)
for d in $T/debs/*.deb; do x=$(mktemp -d); (cd $x && ar x $d && tar -xf data.tar.* -C $T/root); rm -rf $x; done
printf '#!/bin/bash\nexport LD_LIBRARY_PATH=%s:%s${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\nexec %s "$@"\n' \
    $T/root/usr/lib/x86_64-linux-gnu $T/root/usr/local/lib "$(ls $T/root/usr/bin/ocloc-* | head -1)" > $T/bin/ocloc
chmod +x $T/bin/ocloc
pip install "cmake>=3.26" ninja "packaging>=24.2" "setuptools>=77.0.3,<80.0.0" "setuptools-scm>=8" wheel regex \
    jinja2 psutil build scipy
PATH=$T/bin:$PATH SYCL_HOME=$CMPLR_ROOT MAX_JOBS=32 VLLM_VERSION_OVERRIDE=0.1.15.4 \
VLLM_XPU_AOT_DEVICES=pvc VLLM_XPU_XE2_AOT_DEVICES=pvc VLLM_XPU_ENABLE_XE3P=0 \
    setsid nohup pip install --no-build-isolation --no-deps -v . > build.log 2>&1 &
```

A rerun reuses `build/`.

### 4. vLLM

```bash
cd $VLLM_DIR && VLLM_TARGET_DEVICE=xpu VLLM_REQUIRE_RUST_FRONTEND=1 pip install -e . --no-deps --no-build-isolation -v
```

`Failed to import from vllm._C` is normal on XPU.

### 5. Monarch (~20-30 min)

```bash
git clone https://github.com/meta-pytorch/monarch.git $MONARCH_DIR && cd $MONARCH_DIR && git checkout a0c586486
GCC=/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/gcc-13.4.0-22p27zg/bin
LLVM=/opt/aurora/26.181.0/spack/unified/1.1.1/install/linux-x86_64/llvm-develop-git.87717bf-7gy6r7k
pip install "setuptools-rust>=1.9.0"
CARGO_BUILD_JOBS=16 RUSTC_BOOTSTRAP=1 USE_TENSOR_ENGINE=0 \
PROTOC=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))")/bin/protoc \
CC=$GCC/gcc CXX=$GCC/g++ CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1" CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=$GCC/gcc \
LIBCLANG_PATH=$LLVM/lib BINDGEN_EXTRA_CLANG_ARGS="-I$LLVM/lib/clang/23/include" \
    pip install -e . --no-deps --no-build-isolation
pip install pyzmq pyarrow requests numpy pyre-extensions "typing-extensions>=4.12" cloudpickle lark tabulate \
    opentelemetry-api clusterscope "flask>=2.0" xxhash py-spy aiohttp pytest
python -c "from monarch.actor import Actor; from monarch.job import PBSJob; import monarch._rust_bindings"
```

The variables select GNU gcc 13.4 and libclang from Aurora's spack and torch's `protoc`. Rebuild Monarch after any
torch change. pip's `nvidia-ml-py` warning is harmless on XPU.

### 6. TorchStore

```bash
git clone -b xpu-all https://github.com/songhappy/torchstore.git $TORCHSTORE_DIR
cd $TORCHSTORE_DIR && git checkout 3edc47f
pip install -e . --no-deps --no-build-isolation && pip install pygtrie portpicker
```

`--no-deps`: TorchStore pins an older `torchmonarch`.

### 7. TorchTitan deps

TorchTitan runs from the source tree; install its requirements without `torchmonarch`, holding torch:

```bash
cd $TORCHTITAN_DIR
grep -v "^torchmonarch" torchtitan/rl/requirements.txt > /tmp/rl_reqs.txt
pip install -r requirements.txt -r /tmp/rl_reqs.txt torchdata==0.11.0 tyro==1.0.15 matplotlib \
    --constraint <(echo "torch==2.15.0+xpu")
```

The CUDA-only packages `attn-gym` brings are unused on XPU.

### Reference versions (`pip list` of the validated env)

| Component | Version | Commit | Install |
|---|---|---|---|
| torchtitan | source tree | this branch | -- |
| Monarch (`torchmonarch`) | 0.8.0.dev0 | `a0c586486` | editable Rust build (step 5) |
| TorchStore | 0.0.0.dev0 | `3edc47f` | editable, `--no-deps` (step 6) |
| vLLM | 0.30.1rc1.dev523+g083060d04.xpu | `083060d04` | editable (step 4) |
| vllm-xpu-kernels | 0.1.15.4 | `release/0.1.15.4` | from source (step 3) |
| torch / torchvision | 2.15.0+xpu / 0.29.2+xpu | -- | wheels (step 2) |
| triton-xpu / oneccl / intel-sycl-rt / oneAPI | 3.9.0 / 2022.1.2 / 2026.1.2 / 2026.1 | -- | with torch; oneAPI module |

Others: python 3.12, transformers 5.22.2, datasets 4.7.0, huggingface_hub 1.33.0, spmd_types 0.3.1, torchdata
0.11.0, tyro 1.0.15, renderers 0.1.12.

### Check the env on compute nodes

Run the smoke ([Validation](#validation) 1) before anything else: it proves vLLM's kernels (real completions, non-zero reward),
Monarch across hosts and TorchStore across nodes.

## Running

One PBS script for every node count; submit from the checkout (it runs that tree) and pass overrides with `-v`:

```bash
qsub -l select=2 -q debug -v ARM=full,PPN=4,NUM_STEPS=2 scripts/rl/xpu/run_grpo.sh
qsub -l select=16 -q capacity -v ARM=lora,PPN=12,NUM_STEPS=2 scripts/rl/xpu/run_grpo.sh
qsub -l select=16 -l walltime=01:00:00 -q capacity -v CONFIG=rl_grpo_qwen3_1_7b_flex_16n_ppn8_gen_c8_10steps \
    scripts/rl/xpu/run_grpo.sh
```

| Override | Default | Meaning |
|---|---|---|
| `ARM` / `PPN` / `NUM_STEPS` | `full` / `4` / `10` | Qwen3-0.6B arm (`full`, `lora`); tiles per node (1 or even, <= 12); 2, 10 or 200 steps |
| `CONFIG` | from the above | any config in `alphabet_sort.py` (larger models) |
| `CONDA_ENV`, `DUMP_FOLDER`, `LOG` | `monarch_t215`, `outputs/<config>`, `train.log` there | env and outputs |

Queues (per user): `debug` (<= 2 nodes) and `debug-scaling` 1 running + 1 queued; `capacity` (<= 16 nodes) 2
running, 5 total.

## A healthy run

- every requested step logs `Train | Step: N` (larger models: still training at walltime also passes);
- reward non-zero from step 1 (an occasional `Consecutive untrainable batches: 1/10` is fine; climbing to 10 means
  the generators got zero weights);
- `grad_norm` (Qwen3-0.6B): LoRA ~0.01-0.8, full ~0.1-5;
- the `Mesh split` line in `train.log` shows the expected trainer and generator meshes (multi-node runs).

## Validation

Four sets of runs, in order, on the env you built. Each run must reach its steps; judge it by [A healthy
run](#a-healthy-run) and the expected meshes below; the `Mesh split` line in `train.log` shows both (1-node runs print
none: their layout is fixed). Model: Qwen3-0.6B (sets 1-3), alphabet sort; `<arm>` = `full` and `lora` (each row is
two runs). Trainer mesh = `dp_replicate x dp_shard` (FSDP) unless it says TP; generator mesh = vLLM engines x tiles
per engine (tiles per engine = the engine's TP).

### 1. Smoke (4 runs, ~30 min)

| Runs | Submit (`qsub ... scripts/rl/xpu/run_grpo.sh`) | Steps | Trainer mesh | Generator mesh |
|---|---|---|---|---|
| 1 node: 2 trainer + 2 generator tiles | `-l select=1 -v ARM=<arm>,NUM_STEPS=2` | 2 | 1 x 2 | 2 x 1 (same node) |
| 8 nodes x 4 tiles | `-l select=8 -v ARM=<arm>,NUM_STEPS=2` | 2 | 4 x 4 (nodes 1-4) | 16 x 1 (nodes 5-8) |

### 2. Basic regression (18 runs, ~2 h plus queue)

| Runs | Submit | Steps | Trainer mesh | Generator mesh |
|---|---|---|---|---|
| 2 nodes x P tiles, P = 1, 2, 4, 8, 12 | `-l select=2 -v ARM=<arm>,PPN=<P>,NUM_STEPS=2` | 2 | 1 x P (node 1) | P x 1 (node 2) |
| 16 nodes x P, P = 4, 12 | `-l select=16 -q capacity -v ARM=<arm>,PPN=<P>,NUM_STEPS=2` | 2 | 8 x P (nodes 1-8) | 8P x 1 (32 / 96; nodes 9-16) |
| 1 node, 10 steps | `-l select=1 -v ARM=<arm>` | 10 | 1 x 2 | 2 x 1 (same node) |
| 8 nodes x 4, 10 steps | `-l select=8 -v ARM=<arm>` | 10 | 4 x 4 (nodes 1-4) | 16 x 1 (nodes 5-8) |

### 3. Generator scaling (12 runs, ~40 min)

One vLLM engine grows by tensor parallelism; the trainer stays one copy on node 1. TP must divide the model's 8 KV
heads, so an engine spans nodes only on 4-tile nodes.

| Runs | Submit | Steps | Trainer mesh | Generator mesh |
|---|---|---|---|---|
| 2 nodes x 12, engine TP A, A = 1, 2, 4, 8 | `-l select=2 -v ARM=<arm>,PPN=12,GEN_TILES=<A>,NUM_STEPS=2` | 2 | 1 x 12 (node 1) | 1 x A (node 2) |
| 3 nodes x 4, one engine over 2 nodes | `-l select=3 -v ARM=<arm>,PPN=4,GEN_NODES=2,NUM_STEPS=2` | 2 | 1 x 4 (node 1) | 1 x 8 (nodes 2-3) |
| 9 nodes x 4, four such engines | `-l select=9 -q capacity -v ARM=<arm>,PPN=4,GEN_NODES=2,GEN_ENGINES=4,NUM_STEPS=2` | 2 | 1 x 4 (node 1) | 4 x 8 (nodes 2-9) |

### 4. Larger models (2 runs, 2 and 16 nodes)

Link the model under `assets/hf/` first (`ln -sfn /lus/flare/projects/Aurora_deployment/intel/models/<name> assets/hf/<name>`),
then submit with a 1 h walltime (`-l walltime=01:00:00`) and `<dir>` on your project's `/lus/flare` space:
`rl_grpo_qwen3_14b_flex` (upstream's recipe) saves a full checkpoint at step 10, ~166 GB, which overfills a home
quota. Delete `<dir>/checkpoint` afterwards.

| Runs | Submit | Steps | Trainer mesh | Generator mesh |
|---|---|---|---|---|
| Qwen3-14B, upstream `rl_grpo_qwen3_14b_flex`, 2 nodes | `-l select=2 -q capacity -v CONFIG=rl_grpo_qwen3_14b_flex,DUMP_FOLDER=<dir>` | 10 | TP 8 (node 1) | 1 x 8 (node 2) |
| Qwen3-1.7B `rl_grpo_qwen3_1_7b_flex_16n_ppn8_gen_c8_10steps`, 16 nodes | `-l select=16 -q capacity -v CONFIG=rl_grpo_qwen3_1_7b_flex_16n_ppn8_gen_c8_10steps,DUMP_FOLDER=<dir>` | 10 | 8 x 8 (nodes 1-8) | 8 x 8 (nodes 9-16) |

### What to report

Per run: config, nodes x tiles, steps reached, mesh, pass / fail, and the `train.log` path of any failure; any build
step that failed or needed a change; `pip list` of the env.

## XPU settings and why

**oneCCL workarounds** (on by default for XPU): TorchStore volumes run in their own processes
(`storage_volumes_in_own_procs`, oneCCL [#222](https://github.com/uxlfoundation/oneCCL/issues/222): otherwise
cross-node broadcasts deliver nothing), and the first weight pull runs one generator at a time
(`sequential_initial_pull`, [#225](https://github.com/uxlfoundation/oneCCL/issues/225)). Do not set
`CCL_ENABLE_SYCL_KERNELS=0` (FSDP segfaults).

**Environment the launcher sets:**

| Setting | Why |
|---------|-----|
| `CCL_OP_SYNC` unset | makes large cross-node `all_reduce` a silent no-op ([#223](https://github.com/uxlfoundation/oneCCL/issues/223)) |
| `CCL_WORKER_AFFINITY` unset | oneCCL 2022 aborts communicator init on an explicit list |
| `CCL_SYCL_BROADCAST_SCALEOUT_THRESHOLD=0` | #222: multi-tile generators otherwise get no weights (completions `!!!!`) |
| `CCL_SYCL_SCALEOUT_HOST_BUF_SIZE=256 MiB` | the 2 GiB default fills host memory at 12 tiles per node |
| `SYCL_UR_USE_LEVEL_ZERO_V2=1` | the v1 Level Zero adapter aborts about once per 100-150 steps |
| `ZE_FLAT_DEVICE_HIERARCHY=FLAT` (`env_torch215.sh`) | one device per tile; the oneAPI 2026.1 module sets COMPOSITE |
| thread caps (`env_torch215.sh`) | PBS allows 8192 tasks per node |
| no `TORCHINDUCTOR_CACHE_DIR`, `TMPDIR=/tmp` | node-local compile cache (a shared one on `/home` races) |
| `unset HOSTNAME` | otherwise every node looks local to TorchStore |
| `mpiexec --cpu-bind=none` | PALS otherwise pins each worker to one core |
| `ulimit -c 0` | teardown aborts would write ~10 GB cores |

## Known issues

- **A finished run exits with status 1** (teardown abort after the last step; `undeliverable ... generate()` after
  `Closing:` too). Judge a run by its step lines.
- **Larger models: the logprob-parity max can exceed 10** (open). The generator's and trainer's log-probs differ by
  more than 10 at the worst token on the larger models while the mean stays small and training completes; the
  regression report then flags the run on that check. Report the `bit_wise/` metrics; the cause is not yet known.
- **Two trainer copies spanning nodes hang** (FSDP2 HSDP on XPU, also without RL). Use one copy per node or one over
  all trainer nodes.
- **128 oneCCL communicators per process**: keep one copy's `dp_shard` under ~100 (beyond: `code 11`).
- **The first weight pull is slow at large `dp_shard`** (~40 s per generator at 1 x 96): allow walltime.
- **Hung run, no error in `train.log`**: one trainer rank raised;
  `grep -l forward_backward_error $DUMP_FOLDER/structured_logs/rl_trainer*`.
- **Checkpoint saving is off** (`load_only=True`; the save hung at 16 nodes). Do not disable the checkpointer: it also
  loads the pretrained weights.
- **Upstream's larger-model recipes save a full checkpoint at step 10** (`rl_grpo_qwen3_14b_flex`: ~166 GB,
  `rl_grpo_qwen3_1_7b_flex`: ~41 GB): set `DUMP_FOLDER` on `/lus/flare`, not under `$HOME`, and delete the checkpoint
  after the run. The 16-node layouts save nothing.
- **vLLM "not enough free memory" at startup**: usually a stale memory report; resubmit.
