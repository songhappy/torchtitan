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

Run the smoke (experiment 1) before anything else: it proves vLLM's kernels (real completions, non-zero reward),
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
| `TRN_COPIES`, `GEN_TP`, `TRN_TILES`, `TRN_NODES`, `GEN_TILES`, `GEN_NODES`, `GEN_ENGINES`, `BATCH_SCALED` | unset | layout variants used by the experiments below |
| `CONFIG` | from the above | any config in `alphabet_sort.py` (larger models) |
| `CONDA_ENV`, `DUMP_FOLDER`, `LOG` | `monarch_t215`, `outputs/<config>`, `train.log` there | env and outputs |

Queues (per user): `debug` (<= 2 nodes) and `debug-scaling` 1 running + 1 queued; `capacity` (<= 16 nodes) 2
running, 5 total.

## A healthy run

- every requested step logs `Train | Step: N` (larger models: still training at walltime also passes);
- reward non-zero from step 1 (an occasional `Consecutive untrainable batches: 1/10` is fine; climbing to 10 means
  the generators got zero weights);
- `grad_norm` (Qwen3-0.6B): LoRA ~0.01-0.8, full ~0.1-5;
- the `Mesh split` line in `train.log` shows the expected mesh.

## Experiments and expected results

Qwen3-0.6B, alphabet sort, both arms unless noted; mesh = trainer `dp_replicate x dp_shard`. Sections 1-2 and the
marked rows of 7 ran on this env on 2026-11-09 (after 1- and 2-node compute checks passed); the rest ran
2026-10-28 .. 11-04 on the previous nightly stack (same vLLM, TorchStore, oneCCL).

### 1. Smoke (after every stack change)

| Layout | Submit | Expected |
|---|---|---|
| 1 node: 2 trainer + 2 generator tiles | `-l select=1 -v ARM=<arm>,NUM_STEPS=2` | 2/2 steps, mesh 1 x 2 |
| 2 nodes x 4 | `-l select=2 -v ARM=<arm>,NUM_STEPS=2` | 2/2, mesh 1 x 4 |

Result: 4/4 (2026-11-09, this env).

### 2. Basic regression (18 runs; the gate for any code change)

Default layout: trainer on the first half of the nodes, one copy per node; single-tile generators on the rest.

| Runs | Submit | Expected |
|---|---|---|
| 2 nodes x 1, 2, 4, 8, 12 tiles | `-l select=2 -v ARM=<arm>,PPN=<P>,NUM_STEPS=2` | 2/2, mesh 1 x P |
| 16 nodes x 4 and 12 | `-l select=16 -q capacity -v ARM=<arm>,PPN=<P>,NUM_STEPS=2` | 2/2, mesh 8 x P |
| 1 and 8 nodes x 4, 10 steps | `-l select=<1,8> -v ARM=<arm>` | 10/10, mesh 1 x 2 / 4 x 4 |

Result: 18/18 (2026-11-09, this env; peak 2134 of 8192 tasks and 128 GB host memory per node).

### 3. Weight transport at scale

16 nodes x 4 and x 12: one copy per node (8 x 4, 8 x 12) and one over all trainer tiles (`TRN_COPIES=1`, 1 x 96),
2 steps: 6/6.

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

Result: 12/12. TP must divide the KV heads (8), so an engine spans nodes only on 4-tile nodes.

### 6. Both sides at scale

Default layout at 4 and 8 nodes x 12, and `BATCH_SCALED=1` (`NUM_STEPS=10` or `30`) on 2-16 nodes: 12/12.

### 7. Larger models (flex attention; `CONFIG=...`, 10 steps)

Link each model under `assets/hf/` (`/lus/flare/projects/Aurora_deployment/intel/models/<name>`; gpt-oss:
`gpt-oss-20b-BF16`). `_flex` = upstream's recipe unchanged except flex attention (varlen has no XPU kernel).
`_16n_...` = our 16-node layouts (FSDP instead of trainer TP / EP, no checkpoint save).

**Upstream layout** (`rl_grpo_<model>_flex`):

| Model | Upstream layout: trainer / generator | Nodes | Result on XPU |
|---|---|---|---|
| Qwen3-1.7B (dense) | TP 2 / one engine TP 4 (6 GPUs) | 1 | 10/10 (this env, 2026-11-09) |
| Qwen3-14B (dense) | TP 8 / one engine TP 8 (16 GPUs) | 2 | 10/10 (this env, 2026-11-09) |
| Qwen3-30B-A3B (MoE, 128 experts) | FSDP 2 x TP 2 x EP 4 / DP 2 x TP 2 x EP 4 (8 GPUs) | 1 | fail: exits at generator start with oneCCL KVS errors (open) |
| gpt-oss-20b (MoE, 32 experts) | TP 2 x EP 2 / TP 4 x EP 4; 5 prompts per step | 1 | fail: trainer out of memory (20B on 2 tiles of 64 GB) |
| Qwen3.5-9B / Qwen3.6-27B (GDN) | -- | -- | blocked: vLLM's GDN path needs NVIDIA-only CuTe kernels |

**16-node layouts** (first 8 nodes train, last 8 generate; `-l select=16 -q capacity`, walltime 1-2 h):

| Model | Config | Trainer / generator | Result |
|---|---|---|---|
| Qwen3-1.7B | `rl_grpo_qwen3_1_7b_flex_16n_ppn8_gen_c8_10steps` | 8 x 8 FSDP / 8 engines TP 8 | 10/10 (this env, 2026-11-09) |
| Qwen3-14B | `rl_grpo_qwen3_14b_flex_16n_ppn8_gen_c8_10steps` | 8 x 8 FSDP, full activation remat / 8 engines TP 8 | healthy at walltime (~4.5 min/step; previous stack) |
| Qwen3-30B-A3B | `rl_grpo_qwen3_30b_a3b_flex_16n_ppn8_trn_c1_gen_c8_10steps` | 1 x 64 FSDP / 8 engines TP 4 x DP 2, EP 8, vLLM at 0.7 of a tile | 10/10 (this env, 2026-11-09; ~40 min) |
| gpt-oss-20b | `rl_grpo_gpt_oss_20b_flex_16n_ppn4_trn_c1_gen_c8_10steps` | 1 x 32 FSDP (8 nodes x 4) / 8 engines TP 4, EP 4, vLLM at 0.7 | healthy at walltime (previous stack) |

Not retried, and why: MoE with one copy per 8-tile node runs out of memory at step 2; vLLM at the default 0.9 of
a tile fails the first forward (`UR_RESULT_ERROR_OUT_OF_RESOURCES`); Qwen3-1.7B at 1 x 96 exceeds the 30-min gloo
timeout. Rules: FSDP shards divide the FFN width and expert count; TP divides the KV heads; MoE needs one trainer
copy over all trainer nodes and `gpu_memory_limit` 0.7. Two 10/10 runs fail the logprob-parity check
([known issue](#known-issues)).

### 8. Logprob parity (trainer vs generator)

Every run prints `bit_wise/` metrics. Mean log-ratio: Qwen3-0.6B ~-0.0007, Qwen3-30B-A3B ~-0.01 (~7% of tokens
differ). The max exceeds 10 on the larger models ([known issue](#known-issues)).

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
- **Larger models: logprob-parity max > 10** (open). Qwen3-1.7B upstream layout 13.3, Qwen3-30B-A3B at 16 nodes 24.4
  (Qwen3-14B on the previous stack); the mean stays small and training runs 10/10. The report fails these runs on
  that check; they do not crash. Cause unknown (vLLM vs trainer numerics on XPU); report the `bit_wise/` metrics.
- **Two trainer copies spanning nodes hang** (FSDP2 HSDP on XPU, also without RL). Use one copy per node or one over
  all trainer nodes.
- **128 oneCCL communicators per process**: keep one copy's `dp_shard` under ~100 (beyond: `code 11`).
- **The first weight pull is slow at large `dp_shard`** (~40 s per generator at 1 x 96): allow walltime.
- **Hung run, no error in `train.log`**: one trainer rank raised;
  `grep -l forward_backward_error $DUMP_FOLDER/structured_logs/rl_trainer*`.
- **Checkpoint saving is off** (`load_only=True`; the save hung at 16 nodes). Do not disable the checkpointer: it also
  loads the pretrained weights.
- **vLLM "not enough free memory" at startup**: usually a stale memory report; resubmit.
