# GRPO on Intel XPU (Aurora, torch 2.15)

This branch runs end-to-end GRPO on Aurora (Intel Data Center GPU Max 1550) with the upstream torchtitan RL stack --
Monarch actors, TorchStore weight sync, vLLM generation, FSDP2 training -- on **torch 2.15.0 (XPU release) and oneCCL
2022**, on top of upstream torchtitan main. It is written so a validation engineer can rebuild the stack, rerun every
experiment below and check each run against the expected result.

The XPU support follows the CUDA code paths (device checks in the generator, worker and launch env; oneCCL
workarounds that default on for XPU). The Aurora-specific pieces are the launcher (`scripts/rl/xpu/run_grpo.sh`,
Monarch's `PBSJob` the way `SlurmJob` is used on Slurm), the environment script (`scripts/rl/xpu/env_torch215.sh`)
and the layout recipes in `torchtitan_recipes/rl/alphabet_sort.py`.

Contents: [Quick start](#quick-start) - [Environment setup](#environment-setup) - [Running](#running) -
[A healthy run](#a-healthy-run) - [Experiments and expected results](#experiments-and-expected-results) -
[XPU settings and why](#xpu-settings-and-why) - [Known issues](#known-issues). What is still missing on XPU, phase by
phase and repo by repo: [xpu_rl_roadmap.md](xpu_rl_roadmap.md).

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

One conda env, `monarch_t215` (the launcher's default `CONDA_ENV`). Build on a UAN (login node), never on a compute
node. Prerequisites: Aurora access, miniforge3, the model weights on a filesystem the compute nodes read, and the
dataset cached once on the UAN (runs set `HF_HUB_OFFLINE=1`):
`python -c "import datasets; datasets.load_dataset('kalomaze/alphabetic-arxiv-authors-it1', split='train')"`.

| Repo | Branch | Commit | Contents |
|------|--------|--------|----------|
| torchtitan | this branch | -- | upstream main + the XPU RL commits, launcher, recipes, this guide |
| monarch | upstream `main` (github.com/meta-pytorch/monarch) | `a0c586486` | no local patches: the XPU actor runtime (#4307) and `PBSJob` (#5067) are merged |
| torchstore | `xpu-all` (github.com/songhappy/torchstore) | `3edc47f` | upstream main `e500a6f` + PR #171 (XCCL transport), #201 (replicated regions fetched once), the routing fix 16 nodes x 12 tiles needs, accelerator stream sync |
| vLLM | upstream `main` | `083060d04` | no local patches |
| vllm-xpu-kernels | `release/0.1.15.4` | -- | built from source against torch 2.15.0 (the 0.1.15.4 wheel is built for torch 2.14) |

Why vLLM stays at `083060d04` although newer vLLM pins torch 2.15.0: vLLM #60517 (2026-10-08) removed the
`PassConfig` sequence-parallel fields that torchtitan's generator passes (upstream torchtitan main too, as of
2026-11-09), and every vLLM that pins torch 2.15.0 comes after it.

### Changes from the previous version of this guide

| Layer | Previous (torch 2.15 nightly) | Now |
|---|---|---|
| torch / torchvision | 2.15.0.dev20261005 / 0.30.0.dev20261005 (nightly) | **2.15.0 / 0.29.2** (release) |
| triton-xpu | 3.9.0 (nightly build) | 3.9.0 (release) + vLLM's `triton==3.9.0+xpu` placeholder |
| Monarch | fork `songhappy/monarch` `xpu-all` `2f1637ca6` | **upstream `main` `a0c586486`** (no fork) |
| vllm-xpu-kernels | 0.1.15.4 from source vs the nightly | 0.1.15.4 from source vs torch 2.15.0 (same CI ocloc) |
| vLLM, TorchStore | `083060d04`, fork `xpu-all` `3edc47f` | unchanged |
| oneAPI / oneCCL / SYCL runtime | 2026.1 / 2022.1.2 / 2026.1.2 | unchanged (torch 2.15.0 requires the same) |
| transformers / renderers | 5.20.2 / 0.1.11 | 5.22.2 / 0.1.12 |
| Build steps | -- | re-activate the env after step 2 (below) |

```bash
export TORCHTITAN_DIR=$HOME/git/torchtitan    # this branch
export MONARCH_DIR=$HOME/git/monarch TORCHSTORE_DIR=$HOME/git/torchstore VLLM_DIR=$HOME/git/vllm
export KERNELS_DIR=$HOME/git/vllm-xpu-kernels CONDA_PREFIX_BASE=$HOME/miniforge3
```

### 1. Conda env and toolchain

```bash
source $TORCHTITAN_DIR/scripts/rl/xpu/env_torch215.sh   # oneAPI 2026.1, gcc, ZE_FLAT_DEVICE_HIERARCHY=FLAT
$CONDA_PREFIX_BASE/bin/conda create -n monarch_t215 python=3.12 -y
eval "$($CONDA_PREFIX_BASE/bin/conda shell.bash hook)" && conda activate monarch_t215
conda install -y -c conda-forge rust=1.97.1
```

Never source a oneAPI 2025.x environment in the same shell: torch ships the 2026.1 SYCL runtime, and two SYCL
runtimes in one process segfault.

### 2. torch 2.15.0, Triton and vLLM's other requirements

vLLM `083060d04`'s `requirements/xpu.txt` pins torch 2.14; install torch 2.15.0 instead and everything else from that
file:

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

torch 2.15.0 brings triton-xpu 3.9.0, oneccl 2022.1.2 and intel-sycl-rt 2026.1.2. The `oneccl` wheel adds conda
activation hooks that put the env's `lib/` first on `LD_LIBRARY_PATH`; they only take effect on activation, so
without the re-activation `import torch` fails in that shell (`libsycl.so.9: undefined symbol: urGraphGetIdExp`). The
`triton` placeholder keeps xgrammar from pulling stock (CUDA) Triton over `triton-xpu`; if `import triton.language`
still fails, reinstall `triton-xpu` with `--force-reinstall --no-deps`.

### 3. vllm-xpu-kernels from source (~3 h)

Device code is compiled at link time by the first `ocloc` on `PATH`. Aurora's system ocloc (25.18 / IGC 2.11)
miscompiles the flash-attention kernels for PVC **with no error** (vLLM then generates garbage), so use the ocloc the
kernels' CI pins (IGC 2.34.4 + compute-runtime 26.18, listed in `build_script/gpu_runtime_packages.json`), unpacked
from its .deb files without root:

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

`VLLM_XPU_*`: Aurora is PVC; Xe3P is folded into every target list when on, and Aurora's ocloc rejects it. A rerun
reuses `build/`. To re-link with another ocloc, move `build/temp*/*.so` aside and rerun (~7 min).

### 4. vLLM

```bash
cd $VLLM_DIR && VLLM_TARGET_DEVICE=xpu VLLM_REQUIRE_RUST_FRONTEND=1 pip install -e . --no-deps --no-build-isolation -v
```

`VLLM_REQUIRE_RUST_FRONTEND=1` makes a missing `cargo` an error instead of silently dropping `_rust_tool_parser`.
`Failed to import from vllm._C` at import is normal on XPU (that module is the CUDA kernels).

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

Why: `USE_TENSOR_ENGINE=0` builds only the actor runtime RL uses; `RUSTC_BOOTSTRAP=1` lets stable rustc take the
nightly flags `setup.py` passes; `PROTOC` is the one torch bundles; `CC`/`CXX` must be GNU gcc (13.4 tested; the
oneAPI env sets `CC=icx`, whose intrinsics GNU `ld` cannot link); the cargo linker replaces conda's, whose sysroot
lacks the glibc 2.38 symbols `aws-lc-sys` needs; `LIBCLANG_PATH` / `BINDGEN_EXTRA_CLANG_ARGS` let `libsqlite3-sys`
find libclang. Rebuild Monarch after any torch change. pip warns that `torchmonarch` requires `nvidia-ml-py`; it is
not needed on XPU.

### 6. TorchStore

```bash
git clone -b xpu-all https://github.com/songhappy/torchstore.git $TORCHSTORE_DIR
cd $TORCHSTORE_DIR && git checkout 3edc47f
pip install -e . --no-deps --no-build-isolation && pip install pygtrie portpicker
```

`--no-deps` because TorchStore pins an older `torchmonarch`.

### 7. TorchTitan deps

TorchTitan runs from the source tree and is not pip-installed. Install its requirement files minus `torchmonarch`
(step 5 built it), holding torch:

```bash
cd $TORCHTITAN_DIR
grep -v "^torchmonarch" torchtitan/rl/requirements.txt > /tmp/rl_reqs.txt
pip install -r requirements.txt -r /tmp/rl_reqs.txt torchdata==0.11.0 tyro==1.0.15 matplotlib \
    --constraint <(echo "torch==2.15.0+xpu")
```

`attn-gym[linear]` brings CUDA-only packages (`nvidia-cutlass-dsl`, `cuda-bindings`); they are unused on XPU.

### Reference versions (`pip list` of the validated env)

RL stack:

| Component | pip name / version | Commit | Source, install |
|---|---|---|---|
| torchtitan | -- (runs from the source tree) | this branch | github.com/songhappy/torchtitan `rl-xpu-aurora-torch215` |
| Monarch | `torchmonarch` 0.8.0.dev0 | `a0c586486` | github.com/meta-pytorch/monarch `main`, editable Rust build (step 5) |
| TorchStore | `torchstore` 0.0.0.dev0 | `3edc47f` | github.com/songhappy/torchstore `xpu-all`, editable, `--no-deps` (step 6) |
| vLLM | `vllm` 0.30.1rc1.dev523+g083060d04.xpu | `083060d04` | github.com/vllm-project/vllm `main`, editable (step 4) |
| vllm-xpu-kernels | `vllm-xpu-kernels` 0.1.15.4 | `release/0.1.15.4` | built from source against torch 2.15.0 with the CI ocloc (step 3) |

PyTorch and Intel runtime:

| Package | Version | Package | Version |
|---|---|---|---|
| python | 3.12 | oneccl | 2022.1.2 |
| torch | 2.15.0+xpu | intel-sycl-rt | 2026.1.2 |
| torchvision | 0.29.2+xpu | oneAPI (module, `env_torch215.sh`) | 2026.1 |
| triton-xpu / triton | 3.9.0 / 3.9.0+xpu (placeholder) | | |

Python libraries: transformers 5.22.2, datasets 4.7.0, huggingface_hub 1.33.0, spmd_types 0.3.1, torchdata 0.11.0,
tyro 1.0.15, renderers 0.1.12.

### Check the env on compute nodes

Before the experiments, one 1-node smoke in each arm (`qsub -l select=1 -v ARM=full scripts/rl/xpu/run_grpo.sh`,
then `ARM=lora`) and one 2-node smoke (`-l select=2`) must each log their steps; this proves the kernels (vLLM's
completions are real text, the reward is non-zero), Monarch across hosts and TorchStore across nodes.

## Running

`scripts/rl/xpu/run_grpo.sh` is one PBS script for every node count. Submit it from the checkout (it runs the code of
the directory `qsub` ran from) and pass overrides with `qsub -v`:

```bash
qsub -l select=2 -q debug -v ARM=full,PPN=4,NUM_STEPS=2 scripts/rl/xpu/run_grpo.sh
qsub -l select=16 -q capacity -v ARM=lora,PPN=12,NUM_STEPS=2 scripts/rl/xpu/run_grpo.sh
qsub -l select=16 -l walltime=01:00:00 -q capacity -v CONFIG=rl_grpo_qwen3_1_7b_flex_16n_ppn8_gen_c8_10steps \
    scripts/rl/xpu/run_grpo.sh
```

| Override | Default | Meaning |
|----------|---------|---------|
| `ARM` | `full` | `full` or `lora` (Qwen3-0.6B configs) |
| `PPN` | `4` | tiles per node; 1 or an even number up to 12 |
| `NUM_STEPS` | `10` | 2, 10 or 200 |
| `TRN_COPIES` | unset | `_trn_c<R>`: R trainer copies, FSDP across nodes |
| `GEN_TP` | unset | `_gen_tp<T>`: each vLLM generator over T tiles |
| `TRN_TILES`, `TRN_NODES`, `GEN_TILES`, `GEN_NODES`, `GEN_ENGINES`, `BATCH_SCALED` | unset | the one-side layouts (`_trn_t`, `_trn_n`, `_gen_t`, `_gen_n`, `x<E>`, `_batch_scaled`), see [Experiments](#experiments-and-expected-results) |
| `CONFIG` | from the above | any config in `alphabet_sort.py` (the larger models are set this way) |
| `CONDA_ENV` | `monarch_t215` | the env built above |
| `DUMP_FOLDER`, `LOG` | `outputs/<config>`, `$DUMP_FOLDER/train.log` | output locations |
| `TORCHTITAN_DIR`, `ONEAPI_ENV_SCRIPT`, `MPIEXEC`, `EXTRA_ARGS` | see the script | rarely needed |

Queue limits per user: `debug` (<= 2 nodes) and `debug-scaling` one running + one queued, `capacity` (<= 16 nodes)
two running and five in total; past that `qsub` is refused (`would exceed queue generic's per-user limit`).

How it launches: the script dumps a `monarch.job.PBSJob({"hosts": <nodes>})` and points `TORCHTITAN_MONARCH_JOB` at
it; `python -m monarch._src.job._pbs_batch` starts one Monarch worker per node with Cray PALS `mpiexec` and runs
`torchtitan.rl.actors.monarch.job_launcher` as the client, which places the trainer and generators on the job's
hosts and runs `torchtitan.rl.train`.

## A healthy run

- every requested step logs a `Train | Step: N` line ("still training at walltime" counts as a pass for the larger
  models, whose steps take minutes);
- reward is non-zero from step 1. An occasional `Consecutive untrainable batches: 1/10` is normal; reward 0 with that
  count climbing toward 10 means the generators received zero weights;
- `grad_norm` (Qwen3-0.6B): LoRA about 0.01-0.5, full about 0.1-4 (full is over all parameters);
- the `Mesh split` line in `train.log` shows the mesh you expect -- read the mesh from that line, not the flags.

## Experiments and expected results

Sections 1-2 are re-run on this env (torch 2.15.0, Monarch upstream main) on 2026-11-09, after the compute-node
checks (1 and 2 nodes: torch, vLLM generation, the kernels' flash-attention test, cross-node XCCL broadcast, Monarch
remote spawn) all passed; sections 3-8 were measured 2026-10-28 .. 11-04 on the previous torch 2.15 nightly stack
(same vLLM, TorchStore, oneCCL). Model: Qwen3-0.6B, alphabet-sort task, 0/1
reward, `rl_grpo_qwen3_0_6b_{full,lora}_<N>n_ppn<P>[_variant]_<S>steps`, both arms unless noted. Mesh = trainer
`dp_replicate x dp_shard`.

### 1. Smoke (after every stack change)

| Layout | Submit | Expected |
|---|---|---|
| 1 node: 2 trainer + 2 generator tiles | `-l select=1 -v ARM=<arm>,NUM_STEPS=2` | 2/2 steps, mesh 1 x 2 |
| 2 nodes x 4 | `-l select=2 -v ARM=<arm>,NUM_STEPS=2` | 2/2, mesh 1 x 4 |

Result: 4/4 (2026-11-09, this env).

### 2. Basic regression (18 runs; the gate for any code change)

Default layout: the trainer takes the first N/2 nodes, single-tile generators the rest; one model copy per trainer
node (`dp_replicate` = trainer nodes, `dp_shard` = tiles per node). At `P=10`, where 10 cannot shard the FFN (3072),
the recipe picks the smallest `dp_replicate` that can.

| Runs | Submit | Expected |
|---|---|---|
| 2 nodes x 1, 2, 4, 8, 12 tiles | `-l select=2 -v ARM=<arm>,PPN=<P>,NUM_STEPS=2` | 2/2, mesh 1 x P |
| 16 nodes x 4 and 12 | `-l select=16 -q capacity -v ARM=<arm>,PPN=<P>,NUM_STEPS=2` | 2/2, mesh 8 x P |
| 1 and 8 nodes x 4, 10 steps | `-l select=<1,8> -v ARM=<arm>` | 10/10, mesh 1 x 2 / 4 x 4 |

Result: 18/18 (rerun 18/18 on Monarch main + #4307, 11-03).

### 3. Weight transport at scale

16 nodes x 4 and x 12, one copy per node (8 x 4, 8 x 12) and one copy over all 96 trainer tiles (`TRN_COPIES=1` at
`PPN=12`: 1 x 96), 2 steps: 6/6 pass (11-02; 1 x 96 needs TorchStore `3edc47f`'s routing fix, otherwise the first
weight pull never finishes).

### 4. Trainer scaling, one copy (generators: 12 single-tile engines on the last node)

| Submit (`NUM_STEPS=2`) | What | Expected mesh |
|---|---|---|
| `-l select=2 -v PPN=12,TRN_TILES=<A>`, A = 1, 2, 4, 8, 12 | one copy on A tiles of the first node | 1 x A |
| `-l select=<G+1> -v PPN=12,TRN_NODES=<G>`, G = 2, 4, 8 | one copy over G full nodes | 1 x 24 / 48 / 96 |

Result: 16/16. `TRN_COPIES=2` over a subset of nodes (2 x 48) hangs -- [known](#known-issues).

### 5. Generator scaling, one engine (trainer fixed)

| Submit (`NUM_STEPS=2`) | What | Expected |
|---|---|---|
| `-l select=2 -v PPN=12,GEN_TILES=<A>`, A = 1, 2, 4, 8 | 1 x 12 trainer; one vLLM engine of TP A | trains, real completions |
| `-l select=3 -v PPN=4,GEN_NODES=2` | 1 x 4 trainer; one TP-8 engine over 2 nodes | same |
| `-l select=9 -v PPN=4,GEN_NODES=2,GEN_ENGINES=4` | four such engines on 8 nodes | same |

Result: 12/12. Generator TP must divide the model's KV heads (8 for Qwen3-0.6B), so an engine spans nodes only on
4-tile nodes.

### 6. Both sides at scale

The default layout at 4 and 8 nodes x 12 tiles, and `BATCH_SCALED=1` (the batch sized to the trainer;
`NUM_STEPS=10` or `30`, `PPN=4` or `12`) on 2-16 nodes: 12/12.

### 7. Larger models (flex attention; `CONFIG=...`, 10 steps)

Link each model under `assets/hf/` first (`/lus/flare/projects/Aurora_deployment/intel/models/<name>`; gpt-oss uses
`gpt-oss-20b-BF16`). The upstream recipes use varlen attention, which has no XPU kernel; the `_flex` configs keep
each recipe as is (model, seq_len 2048, 8 samples per prompt, prompts per step, `max_tokens` 700, optimizer) and only
rebuild its models on flex attention. The `_16n_...` configs also change the layout to scale out over 16 nodes
(FSDP data parallelism instead of trainer TP / EP) and save no checkpoint.

**Upstream layout** (`rl_grpo_<model>_flex`):

| Model | Upstream layout: trainer / generator | Nodes | Result on XPU |
|---|---|---|---|
| Qwen3-1.7B (dense) | TP 2 / one engine TP 4 (6 GPUs) | 1 | pass |
| Qwen3-14B (dense) | TP 8 / one engine TP 8 (16 GPUs) | 2 | pass |
| Qwen3-30B-A3B (MoE, 128 experts) | FSDP 2 x TP 2 x EP 4 / DP 2 x TP 2 x EP 4 (8 GPUs) | 1 | fail: exits at generator start with oneCCL KVS errors (open) |
| gpt-oss-20b (MoE, 32 experts) | TP 2 x EP 2 / TP 4 x EP 4; 5 prompts per step | 1 | fail: trainer out of memory (20B on 2 tiles of 64 GB) |
| Qwen3.5-9B / Qwen3.6-27B (GDN) | -- | -- | blocked: vLLM's GDN path needs NVIDIA-only CuTe kernels |

**16-node layouts** (first 8 nodes train, last 8 generate; `-l select=16 -q capacity`, walltime 1-2 h):

| Model | Config | Trainer / generator | Result |
|---|---|---|---|
| Qwen3-1.7B | `rl_grpo_qwen3_1_7b_flex_16n_ppn8_gen_c8_10steps` | 8 x 8 FSDP / 8 engines TP 8 | 10/10 |
| Qwen3-14B | `rl_grpo_qwen3_14b_flex_16n_ppn8_gen_c8_10steps` | 8 x 8 FSDP, full activation remat / 8 engines TP 8 | healthy at walltime (~4.5 min/step) |
| Qwen3-30B-A3B | `rl_grpo_qwen3_30b_a3b_flex_16n_ppn8_trn_c1_gen_c8_10steps` | 1 x 64 FSDP / 8 engines TP 4 x DP 2, EP 8, vLLM at 0.7 of a tile | 10/10 |
| gpt-oss-20b | `rl_grpo_gpt_oss_20b_flex_16n_ppn4_trn_c1_gen_c8_10steps` | 1 x 32 FSDP (8 nodes x 4) / 8 engines TP 4, EP 4, vLLM at 0.7 | healthy at walltime |

What fails, and why (so these are not retried): a MoE model with one copy per 8-tile node runs out of device memory
at step 2 when Adam's state appears (~39 GiB gpt-oss, ~57 GiB 30B-A3B per tile); vLLM at the default 0.9 of a tile
fails the first forward with `UR_RESULT_ERROR_OUT_OF_RESOURCES` (loading pulled weights holds two copies); Qwen3-1.7B
at 1 x 96 with 24 TP-4 engines exceeds the 30-min gloo timeout in the sequential first pull.

Layout rules: FSDP shard counts must divide the FFN width and the MoE expert count; generator TP must divide the KV
heads (<= 8); MoE models need one trainer copy over all trainer nodes; MoE engines need `gpu_memory_limit` 0.7.

### 8. Logprob parity (trainer vs generator)

Every regression layout prints the `bit_wise/` metrics (mean log-ratio, differing-token fraction, max). Expected:
Qwen3-0.6B on 2 nodes mean log-ratio ~-0.0007; Qwen3-30B-A3B 1 x 64 ~-0.01 with ~7% of tokens differing. The max
exceeds 10 on 1.7B / 14B / 30B-A3B while the mean stays small (open: not yet judged a failure).

## XPU settings and why

**oneCCL workarounds in the code**, on by default for XPU:

1. **Cross-node broadcast moves no data** ([#222](https://github.com/uxlfoundation/oneCCL/issues/222)). oneCCL fixes
   a process's position on the node at its first communicator; a TorchStore volume inside an FSDP trainer process
   inherits that position, and its broadcasts from tiles 1+ deliver nothing (zero weights, reward 0).
   `Controller.Config.storage_volumes_in_own_procs` runs the volumes in a twin of the trainer mesh.
2. **Concurrent communicator creation fails** ([#225](https://github.com/uxlfoundation/oneCCL/issues/225)).
   `InterGeneratorRouter.Config.sequential_initial_pull` runs the first weight pull one generator at a time; later
   pulls stay concurrent.

Do not set `CCL_ENABLE_SYCL_KERNELS=0`: FSDP training then segfaults.

**Environment the launcher sets:**

| Setting | Why |
|---------|-----|
| `CCL_OP_SYNC` unset | on oneCCL 2022 it turns cross-node `all_reduce` above 4 MiB into a silent no-op ([#223](https://github.com/uxlfoundation/oneCCL/issues/223)) |
| `CCL_WORKER_AFFINITY` unset | oneCCL 2022 aborts communicator init on an explicit list |
| `CCL_SYCL_BROADCAST_SCALEOUT_THRESHOLD=0` | oneCCL #222: otherwise generators over several tiles receive no weights across nodes (every completion `!!!!`) |
| `CCL_SYCL_SCALEOUT_HOST_BUF_SIZE=256 MiB` | each weight-pull communicator holds this host buffer; the 2 GiB default fills host memory at 12 tiles per node |
| `SYCL_UR_USE_LEVEL_ZERO_V2=1` | the v1 Level Zero adapter aborts about once per 100-150 steps |
| `ZE_FLAT_DEVICE_HIERARCHY=FLAT` (`env_torch215.sh`) | one device per tile; the oneAPI 2026.1 module sets COMPOSITE |
| thread caps (`env_torch215.sh`) | Aurora's PBS cgroup allows 8192 tasks per node; at 8-12 tiles per node the default thread pools exceed it |
| no `TORCHINDUCTOR_CACHE_DIR`, `TMPDIR=/tmp` | the compile cache stays per node; a cache on `/home` shared by all nodes races |
| `unset HOSTNAME` | otherwise every node looks local to TorchStore |
| `mpiexec --cpu-bind=none` | PALS otherwise pins each worker and everything it forks to one core |
| `ulimit -c 0` | every teardown aborts and would write ~10 GB of core per process |

## Known issues

- **A finished run exits with status 1.** Teardown aborts after the last `Train | Step` line; the job ends by itself
  within a minute. Judge a run by its step lines, not the PBS exit status. `undeliverable ... generate()` after
  `Closing:` is part of that teardown, not a failure.
- **Two trainer copies across nodes hang** (FSDP2 HSDP on XPU): `dp_replicate=2` with each copy spanning nodes stops
  in the first `forward_backward`, no error, until walltime. It reproduces in plain FSDP2 without RL. Use one copy
  per node or one copy over all trainer nodes.
- **oneCCL allows 128 communicators per process.** TorchStore keeps one per (generator, storage volume); keep one
  copy's `dp_shard` well under 128 (1 x 96 passes, larger fails with `onecclCommInitRank failed with code 11`).
- **The first weight pull grows with `dp_shard`** (sequential, oneCCL #225): ~8 s per generator at 8 x 12, ~40 s at
  1 x 96. Give such runs a long walltime.
- **A hung run with no error in `train.log`** usually means one trainer rank raised and the others wait in FSDP
  collectives: `grep -l forward_backward_error $DUMP_FOLDER/structured_logs/rl_trainer*`.
- **Checkpoint saving is off** (`load_only=True`): the last-step DCP save hung at 16 nodes. Do not disable the
  checkpointer instead; that also skips loading the pretrained weights.
- **vLLM "not enough free memory" at startup** is usually Level Zero's lagging memory report; resubmit first.
- **vLLM with TP > 1 dumps core at exit** after generating correctly; the launcher's `ulimit -c 0` covers it.
