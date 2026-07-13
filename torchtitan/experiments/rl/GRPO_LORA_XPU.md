# Post-Training RL on Intel XPU: Progress and Plan

End-to-end GRPO reinforcement learning on Intel XPU (Borealis cluster) using
the full upstream pipeline: Monarch actors, TorchStore weight sync, vLLM
generation, FSDP2 training.

## Prerequisites

Set these variables to match your local layout before following this guide
(example values are from guoqiong's setup on Borealis):

```bash
export TORCHTITAN_DIR=/home/guoqiong/git/torchtitan
export MONARCH_DIR=/home/guoqiong/git/monarch
export TORCHSTORE_DIR=/home/guoqiong/git/torchstore
export VLLM_DIR=/home/guoqiong/git/vllm
export HF_ASSETS_PATH=/home/guoqiong/models/Qwen3-0.6B       # Qwen3-0.6B weights
export ONEAPI_ENV_SCRIPT=/home/guoqiong/env-3.sh              # oneAPI 2025.3 + gcc-13.3
export CONDA_PREFIX_BASE=/home/guoqiong/miniforge3            # conda installation
```

Requirements:
- Borealis cluster access (UAN for builds, compute nodes for training)
- Intel XPU with 4 tiles (48GB each)
- oneAPI 2025.3 + gcc-13.3 (sourced via `$ONEAPI_ENV_SCRIPT`)
- Conda (miniforge3 or miniconda3)
- Qwen3-0.6B model weights downloaded locally

## Architecture

The pipeline uses Monarch's **actor** framework. A controller orchestrates
two actor groups via async endpoints:

```
+-----------------------------------------------------------------------+
|  Controller (Monarch async main loop)                [TorchTitan]     |
|  Orchestrates: generate prompts -> rollout -> score -> train -> sync  |
+-----------------------------------------------------------------------+
        |                                          |
        | rollout (Monarch async RPC)              | train (Monarch async RPC)
        v                                          v
+-------------------------------+    +-------------------------------+
|  VLLMGenerator                |    |  PolicyTrainer                |
|  XPU 0-1 (DP=2, TP=1)        |    |  XPU 2-3 (FSDP dp_shard=2)   |
|                               |    |                               |
|  Libraries:                   |    |  Libraries:                   |
|  - vLLM (inference engine)    |    |  - TorchTitan (training)      |
|  - vllm_xpu_kernels (attn)   |    |  - PyTorch FSDP2 (DP sync)    |
|  - Monarch (actor runtime)    |    |  - Monarch (actor runtime)    |
|                               |    |  - flex_attention (Triton)    |
|  Features:                    |    |  - LoRA (torchtitan)          |
|  - flash_attn backend         |    |                               |
|  - PagedAttention + KV-cache  |    |  Features:                    |
|  - Continuous batching        |    |  - GRPO loss computation      |
|  - ~53 tok/s generation       |    |  - AdamW optimizer            |
+-------------------------------+    |  - DCP checkpointing          |
                                     +-------------------------------+
        |                                          |
        | pull weights                             | push weights
        v                                          v
+-----------------------------------------------------------------------+
|  TorchStore (weight synchronization)                                  |
|  - CPU-staged shared memory transport (no RDMA on Borealis)           |
|  - xccl collective backend for XPU                                    |
|  - Trainer publishes updated LoRA weights after each step             |
|  - Generator pulls fresh weights before each rollout                  |
+-----------------------------------------------------------------------+

Libraries involved: Monarch, vLLM, TorchStore, TorchTitan, PyTorch (FSDP2,
flex_attention, DCP, inductor/Triton), vllm_xpu_kernels
```

**The 3 components:**

| Component | What it is | Built with | Role |
|-----------|-----------|-----------|------|
| Controller | Plain Python async loop (not a Monarch actor) | TorchTitan | The main loop -- generates prompts, dispatches work to the actors, scores results. Runs on CPU. |
| VLLMGenerator | **Monarch actor** (separate process on XPU 0-1) | vLLM + vllm_xpu_kernels | Generates text completions from prompts (~53 tok/s). Called via `await generator.generate(prompts)`. |
| PolicyTrainer | **Monarch actor** (separate process on XPU 2-3) | TorchTitan + FSDP2 + flex_attention | Computes GRPO loss, updates LoRA weights. Called via `await trainer.train_step(batch)`. |

Monarch spawns the two actors as separate processes on dedicated GPUs and
handles cross-process communication. The controller calls them like normal
async functions -- Monarch makes the process boundary transparent.

Weight sync between them is handled by **TorchStore** (shared memory, <1s per sync).

**Training loop (one step):**

1. **Generate** -- Controller sends prompts to VLLMGenerator, which produces
   completions per prompt
2. **Score** -- Controller evaluates each completion (reward = 0 or 1),
   computes GRPO advantages (better or worse than group average)
3. **Train** -- PolicyTrainer runs forward/backward pass, updates LoRA
   weights via GRPO loss + AdamW
4. **Sync** -- Trainer pushes new weights to TorchStore, generator pulls
   them before next rollout (<1s)
5. **Repeat** -- Model improves each step by learning from its own outputs

---

## Entry Points

All training goes through a single entry point:

```bash
python3 -m torchtitan.experiments.rl.train \
    --module alphabet_sort --config rl_grpo_lora_qwen3_0_6b \
    --hf_assets_path=/path/to/model
```
---

## Repository Details

### vLLM

#### What it is

vLLM is a high-throughput LLM inference engine with continuous batching,
PagedAttention, and tensor parallelism. In the RL pipeline, it serves as
the **generator** -- producing rollout completions from prompts during
each training step. It runs as a Monarch actor on a dedicated GPU mesh.

#### Current status

vLLM already has XPU support in its main branch. Operational notes:

- `cudagraph.enable=False` is required (graph capture not supported on XPU)
- Auto-selected `flash_attn` backend works correctly
- `gpu_memory_limit=0.90` validated for co-location with trainer

#### What has been completed

- Validated vLLM on XPU with flash_attn backend (TP=1 and TP=2)
- Confirmed `vllm-xpu-kernels==0.1.10` wheel works with torch 2.12.0+xpu
- Measured throughput: ~53 tok/s on DP=2 (TP=1)
- Integrated as VLLMGenerator actor in the Monarch pipeline

#### Remaining work

- [ ] Test with larger models (Qwen3-1.7B, Qwen3-4B) on multiple nodes
- [ ] Benchmark TP=2 vs DP=2 throughput (TP=2 was 6.5 tok/s due to xccl overhead)

---

### Monarch

#### What it is

Monarch is Meta's distributed actor framework for orchestrating multi-GPU
workloads. In the RL pipeline, it provides:

- **ProcMesh**: spawns separate process groups for trainer and generator
  on non-overlapping GPU sets
- **Actor endpoints**: async RPC between controller, trainer, and generator
- **Bootstrapping**: environment setup (device affinity, backend init) per
  spawned process
- **HostMesh**: multi-node support via `attach_to_workers` + mesh slicing

#### What has been completed

XPU patches on `xpu-upstream` branch (off `pt/main`):

| File | Change |
|------|--------|
| `device_utils.py` | Replace CUDA scanning with `torch.accelerator.device_count()` |
| `proc_mesh.py` | Add XPU env vars to monitoring; generalize accelerator-init check |
| `job.py` | try/except telemetry imports with stubs |
| `setup.py` | Gate `distributed_sql_telemetry` behind `USE_TENSOR_ENGINE` |
| `test_xpu.py` | 17 tests (6 unit + 11 integration) |

Validation:
- 17/17 tests passing
- All 59 upstream tests pass unmodified
- Full RL pipeline (4 XPUs) running end-to-end

#### Remaining work

- [ ] Submit upstream PR
- [ ] Test multi-node ProcMesh on XPU (via multinode_launcher.py)
- [ ] Address potential review feedback on device-agnostic scope

---

### TorchStore

#### What it is

TorchStore is a distributed key-value store for PyTorch state dicts,
optimized for GPU-to-GPU weight transfer. In the RL pipeline, it provides:

- **Weight publish**: trainer pushes updated model weights after each step
- **Weight pull**: generator pulls fresh weights before each rollout
- **Transport layer**: RDMA (CUDA), shared memory (fallback), or xccl

On Borealis (no RDMA), it uses shared-memory transport with CPU-staged
copies: GPU -> CPU -> shared memory -> CPU -> GPU.

#### What has been completed

XPU patches on `xpu-upstream` branch:

| File | Change |
|------|--------|
| `shared_memory.py` | `pin_memory`/`unpin_memory` no-op on XPU; XPU sync |
| `gloo.py` | Register Gloo backend for XPU devices |
| `torchcomms/cache.py` | Accept any non-CPU device for indexing |
| `xccl.py` (NEW) | Full xccl transport implementation |

Validation:
- TorchStore roundtrip test (push/pull state dict): PASS
- `torchstore_rl.py` example (2 XPU): PASS
- `torchstore_spmd.py` example (2 XPU): PASS
- Full pipeline weight sync (trainer -> generator): PASS

#### Remaining work

- [ ] Submit upstream PR
- [ ] Add pytest for xccl transport
- [ ] Test cross-node transport (shared_memory is single-node only)

---

### TorchTitan (train.py)

**What `train.py` uses:**

| Component | Purpose | XPU Status |
|-----------|---------|------------|
| Monarch ProcMesh | Spawn trainer + generator on separate GPUs | Working |
| vLLM | Fast inference for rollout generation | Working |
| TorchStore | Weight sync: trainer pushes, generator pulls | Working |
| FSDP2 | Data-parallel gradient sync inside trainer | Working |
| flex_attention | Compiled attention kernels with block masks (training) | Working (max_autotune=False) |
| vLLM flash_attn | Inference attention with KV-cache (generation) | Working (vllm_xpu_kernels) |
| LoRA adapters | Parameter-efficient fine-tuning (<1% params) | Working |
| DCP checkpoint | Distributed checkpoint save/load | Working |
| ConfigManager | CLI config parsing (tyro-based) | Working |

---

## XPU-Specific Configuration

| Setting | XPU Value | CUDA Default | Reason |
|---------|-----------|--------------|--------|
| `TORCHINDUCTOR_MAX_AUTOTUNE` | `0` | `1` | Backward configs exceed XPU register limits |
| `cudagraph.enable` | `False` | `True` | No XPU graph support in torch 2.12 |
| `trainer.parallelism` | `dp_shard=2, TP=1` | varies | TP triggers OUT_OF_RESOURCES on XPU |
| `renderer.enable_thinking` | `True` | `False` | False triggers XPU flex_decoding codegen bug |
| `gpu_memory_limit` | `0.9` | `0.9` | Same as upstream (0.6B model fits easily) |
| `expandable_segments` | **NOT SET** | `True` | Breaks oneDNN memory allocator on XPU |
| `ZE_AFFINITY_MASK` | `0,1,2,3` | N/A | Selects XPU tiles |

---

## Environment Setup (Reproducible Recipe)

All builds must run on the UAN (login node), not compute nodes.

### Step 1: Create the conda env

```bash
source $ONEAPI_ENV_SCRIPT   # oneAPI 2025.3, gcc-13.3, XPU build flags
$CONDA_PREFIX_BASE/bin/conda create -n monarch python=3.12 -y
conda activate monarch
```

### Step 2: Install PyTorch XPU

```bash
pip install torch==2.12.0+xpu torchaudio==2.11.0+xpu torchvision==0.27.0+xpu \
    --index-url https://download.pytorch.org/whl/xpu
pip install triton-xpu==3.7.1
```

### Step 3: Install TorchStore

```bash
cd $TORCHSTORE_DIR
git checkout xpu-upstream   # PR: https://github.com/meta-pytorch/torchstore/pull/171
pip install -e . --no-deps --no-build-isolation
pip install pygtrie portpicker
```

`--no-deps` is required -- the `torchmonarch==0.4.1` pin in setup.cfg
conflicts with our editable Monarch install.

### Step 4: Install Monarch

```bash
cd $MONARCH_DIR
git checkout xpu-upstream   # PR: https://github.com/meta-pytorch/monarch/pull/4307
pip install -e python/ --no-deps --no-build-isolation
```

No Rust build needed -- the Python-only install covers the actor runtime
(Layer 1). If you see `ModuleNotFoundError: monarch.distributed_telemetry`,
add a try/except around that import in `python/monarch/_src/job/job.py`.

### Step 5: Install vLLM + XPU kernels

```bash
cd $VLLM_DIR
git checkout main
pip install -e . --no-deps --no-build-isolation

# vllm-xpu-kernels 0.1.10 -- installed from GitHub release wheel
pip install https://github.com/vllm-project/vllm-xpu-kernels/releases/download/v0.1.10/vllm_xpu_kernels-0.1.10-cp38-abi3-manylinux_2_28_x86_64.whl
```

The vllm_xpu_kernels wheel must match torch==2.12.0+xpu. If you see
`RuntimeError: Device string must not be empty`, the kernel wheel was built
against a different torch -- reinstall both in lockstep.

### Step 6: Install Monarch dependencies

Monarch declares these in `pyproject.toml` but since we install with
`--no-deps`, they must be installed manually:

```bash
pip install pyzmq pyarrow requests numpy pyre-extensions "typing-extensions>=4.12" \
    cloudpickle lark tabulate opentelemetry-api clusterscope "flask>=2.0" \
    xxhash py-spy aiohttp
```

### Step 7: Install HF / training deps

```bash
pip install transformers==5.9.0 datasets==4.7.0 tokenizers safetensors \
    tyro einops pillow sentencepiece protobuf huggingface_hub \
    tensorboard wandb tqdm \
    --constraint <(echo "torch==2.12.0+xpu")
```

The `--constraint` prevents pip from pulling a non-XPU torch as a transitive dep.

### Step 8: Install TorchTitan

```bash
cd $TORCHTITAN_DIR
git checkout xpu-upstream   # PR: https://github.com/pytorch/torchtitan/pull/3890
pip install -e . --no-deps --no-build-isolation
```

### Runtime: full script to run a single-node training

Copy-paste this on a compute node (e.g. `ssh <compute-node>`):

```bash
#!/bin/bash
set -e

# Environment
source $ONEAPI_ENV_SCRIPT
eval "$($CONDA_PREFIX_BASE/bin/conda shell.bash hook)"
conda activate monarch

# XPU runtime env vars
export ZE_AFFINITY_MASK=0,1,2,3
export FI_PROVIDER=tcp
export CCL_ATL_OFI_PROVIDER=tcp
export TORCHINDUCTOR_CACHE_DIR=~/.cache/torchinductor_xpu
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Run GRPO+LoRA training (4 XPU tiles: 2 generator + 2 trainer)
cd $TORCHTITAN_DIR
python3 -m torchtitan.experiments.rl.train \
    --module alphabet_sort --config rl_grpo_lora_qwen3_0_6b \
    --hf_assets_path=$HF_ASSETS_PATH
```

Or simply run the launcher scripts:

```bash
# Single node (4 XPU tiles):
cd $TORCHTITAN_DIR/torchtitan/experiments/rl && bash run_grpo_lora_xpu.sh

# Multi-node via PBS (2+ nodes):
cd $TORCHTITAN_DIR/torchtitan/experiments/rl && qsub run_grpo_lora_multinode.sh
```

### Version summary (validated 2026-07-07)

| Package | Version | Source |
|---------|---------|--------|
| torch | 2.12.0+xpu | pytorch.org/whl/xpu |
| triton-xpu | 3.7.1 | pip |
| vllm | editable | $VLLM_DIR main |
| vllm-xpu-kernels | 0.1.10 | GitHub release wheel |
| torchmonarch | editable | $MONARCH_DIR xpu-upstream |
| torchstore | editable | $TORCHSTORE_DIR xpu-upstream |
| torchtitan | editable | $TORCHTITAN_DIR xpu-upstream |
| transformers | 5.9.0 | pip |
| datasets | 4.7.0 | pip |

---

## Known Issues and Fixed Bugs

### Active issues

1. **Inductor cache corruption** -- `CompiledFxGraph has no attribute
   compiled_fn_runner`. Fix: `rm -rf ~/.cache/torchinductor_xpu/`

2. **max_autotune backward crash** -- `TORCHINDUCTOR_MAX_AUTOTUNE=1`
   triggers error 40 (UR_RESULT_ERROR_OUT_OF_RESOURCES).
   Fix: `TORCHINDUCTOR_MAX_AUTOTUNE=0` (set in launch scripts).

3. **Exit code 139 (SIGSEGV)** -- benign oneCCL teardown crash. Run succeeded
   if you see step metrics.

4. **TP=2 on trainer** -- OUT_OF_RESOURCES during lm_head all_gather.
   Use FSDP dp_shard instead.

5. **flex_decoding autotune cold start** -- first run compiles ~75s per
   (Q_LEN, KV_LEN) shape. Dozens of shapes = tens of minutes on cold cache.
   Not a hang -- watch for `SingleProcess AUTOTUNE benchmarking` log lines.
   Fix: pin `TORCHINDUCTOR_CACHE_DIR=~/.cache/torchinductor_xpu` so reboots
   don't wipe the cache.

6. **Monarch atexit TimeoutError** -- `shutdown_context().get(timeout=1.0)`
   fires after successful completion. Benign; no impact on results.

### Fixed bugs

7. **expandable_segments corrupts XPU allocator** (fixed 2026-07-08) --
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` corrupts the SYCL/Level
   Zero memory allocator on XPU. Tensors produce USM pointers that oneCCL
   classifies as "unknown", causing all collectives to fail with "invalid usm
   pointer type". Fix: `train.py` guards the env var behind
   `if "ZE_AFFINITY_MASK" not in os.environ`.

8. **FusedQKVLinear + TorchStore zero-reward bug** (fixed 2026-07-01) --
   `FusedQKVLinear._split_qkv_on_save` hook creates `.contiguous()` copies
   for state_dict. TorchStore's `get_state_dict` writes into these copies,
   but they don't share storage with the actual fused `wqkv` parameter.
   Generator never received updated weights -> zero reward. Fix: call
   `model.load_state_dict(model_sd, strict=False)` after `ts.get_state_dict`
   to trigger `_merge_qkv_on_load`. Also fixed LoRA target_modules from
   `["wq", "wkv", "wo"]` to `["wqkv", "wo"]`.

9. **FSDP collective deadlock in generation** (fixed 2026-06-18) -- when
   ranks emit EOS at different positions, breaking early races
   `dist.all_gather(rewards)` against another rank's FSDP all_gather on the
   same xccl process group. Fix: `dist.all_reduce(done_t, MIN)` per token;
   only break when every rank has emitted EOS.

10. **Eager flex_attention OOM on large seq_len** (fixed 2026-07-06) -- eager
    fallback materializes full `[B,H,L,L]` score matrix. With seq_len=2048
    this OOMs. Fix: use compiled flex_attention (Triton tiled kernel) which
    never materializes the full matrix. The earlier "seq_len limit" was a
    misdiagnosis -- compiled mode handles seq_len=2048 fine.

11. **GPU overlap between trainer and generator** (fixed 2026-07-06) --
    `PerHostProvisioner` set `CUDA_VISIBLE_DEVICES` but XPU ignores this.
    Without `ZE_AFFINITY_MASK` isolation, both processes shared the same
    tiles, causing OOM. Fix: provisioner now sets `ZE_AFFINITY_MASK` when
    XPU is detected.

12. **Monarch telemetry import crash** (fixed 2026-06-09) -- actors-only
    builds (no Rust tensor engine) crash on `from monarch._src.job.process
    import ProcessJob` because `monarch_distributed_telemetry` submodule
    isn't compiled. Fix: try/except around telemetry imports in `job.py`
    with stub that raises only if telemetry is actually called.

13. **vllm_xpu_kernels torch version mismatch** -- `RuntimeError: Device
    string must not be empty` means the kernels wheel was built against a
    different torch. The C++ ABI is unstable across versions. Fix: ensure
    `vllm-xpu-kernels` version matches `torch` exactly (currently both
    target torch 2.12.0+xpu).

---

## Performance (2026-07-07 sweep)

| Config | Throughput (tok/s) | Notes |
|--------|-------------------|-------|
| Baseline (DP=2 gen, FSDP=2 train) | 4,140 | Single node, 4 tiles |
| TP=2 generator | 2,298 | xccl overhead dominates |
| Batch=4 | 3,222 | Memory pressure |

---

## Overall Project Status

**Status: Pipeline functional with real RL signal.**

Validated 2026-07-07: mean reward tracks upward across steps with
AlphabetSort task using Qwen3-0.6B + LoRA. Full upstream architecture
(Monarch + TorchStore + vLLM + FSDP2) runs end-to-end on 4 Intel XPUs.

### Remaining Milestones

| # | Milestone | Priority | Status |
|---|-----------|----------|--------|
| 1 | Non-zero reward | P0 | DONE (mean_r=0.6, 2026-06-18) |
| 2 | Submit Monarch PR | P1 | [PR #4307](https://github.com/meta-pytorch/monarch/pull/4307) open |
| 3 | Submit TorchStore PR | P1 | [PR #171](https://github.com/meta-pytorch/torchstore/pull/171) open |
| 4 | ChunkedLoss parity with CUDA | P1 | Next |
| 5 | max_autotune investigation | P2 | Next |
| 6 | Multi-node scale-out (8+ XPUs) | P2 | Launcher ready |
| 7 | TP=2 trainer perf | P2 | Blocked on xccl overhead |

### Key Risks and Blockers

| Risk | Impact | Mitigation |
|------|--------|------------|
| Monarch/TorchStore PR review delays | Blocks upstream | Submit early, engage reviewers |
| XPU max_autotune not fixable | ~30% perf gap vs CUDA | Coordinate descent only; file torch issue |
| Multi-node xccl transport untested | May not scale beyond 1 node | Test with multinode_launcher.py |
| vllm_xpu_kernels version coupling | Breaks on torch upgrades | Pin versions; test upgrade path |
| TP on trainer broken | Limits scale-out parallelism | FSDP dp_shard works; TP fix is upstream torch issue |

---

## Validation History

| Date | Component | Result |
|------|-----------|--------|
| 2026-06-17 | TorchStore roundtrip (2 XPU) | PASS |
| 2026-06-18 | GRPO pipeline (4 XPU, Monarch actors) | PASS, mean_r=0.60 |
| 2026-06-24 | vLLM backend (4 XPU) | PASS |
| 2026-06-29 | Compiled flex_attention (max_autotune=False) | PASS |
| 2026-07-07 | Perf sweep (baseline/TP=2/batch=4) | PASS, 4140 tok/s baseline |
