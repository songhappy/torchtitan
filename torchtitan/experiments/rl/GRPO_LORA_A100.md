# TorchTitan RL (GRPO + LoRA) -- Polaris A100 Validation Guide

Reproducible end-to-end recipe: clean machine to running GRPO+LoRA on
a single Polaris node with 4+ A100-40GB GPUs (CUDA driver 12.8, cu128 wheels).

---

## 1. Prerequisites

- Polaris login access + `/lus/grand/projects/Intel` project membership
- Miniconda at `~/miniconda3`
- Rust toolchain (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- `~/git/torchtitan` checked out (this repo)
- Hardware: 4x A100-40GB per node (Polaris compute nodes)
- Home quota is 50 GB -- keep large builds on grand (see step 3)

---

## 2. Build the conda environment

Run on a **login node** (needs internet). Install order matters.

```bash
conda create -p /lus/grand/projects/Intel/$USER/envs/monarch-cu128 python=3.12 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lus/grand/projects/Intel/$USER/envs/monarch-cu128

pip install torchmonarch==0.4.1
pip install --no-deps "git+https://github.com/meta-pytorch/torchstore.git@main"
pip install pygtrie portpicker
pip install "git+https://github.com/PrimeIntellect-ai/renderers.git@main"
pip install fastokens prime-pydantic-config
pip install torch==2.12.0.dev20260408+cu128 \
    torchvision==0.27.0.dev20260407+cu128 \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu128
pip install triton==3.5.1 --no-deps   # 3.7 flex kernels exceed A100 shared memory

cd ~/git/torchtitan
pip install -r requirements.txt
pip install datasets transformers tokenizers
```

Verify:

```bash
python -c "
import torch, monarch.actor, torchstore, renderers
print('torch:', torch.__version__, 'cuda', torch.version.cuda)
print('monarch: OK'); print('torchstore: OK')
print('renderers:', renderers.__version__)
"
```

---

## 3. Build vLLM from source

vLLM only publishes cu130 wheels; cu128 requires a source build.
The build must run on a **compute node** (login nodes OOM-kill nvcc).
Keep the repo on grand to avoid home quota overflow.

```bash
# On login node (needs network):
git clone https://github.com/vllm-project/vllm.git \
    /lus/grand/projects/Intel/$USER/vllm
ln -s /lus/grand/projects/Intel/$USER/vllm ~/git/vllm

# Get a compute node:
qsub -I -A Intel -q preemptable -l select=1:ncpus=64 \
    -l walltime=02:00:00 -l filesystems=home:grand
```

On the compute node:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /lus/grand/projects/Intel/$USER/envs/monarch-cu128
module load gcc-native/13
module load cuda/12.9

export CC=/opt/cray/pe/gcc-native/13/bin/gcc
export CXX=/opt/cray/pe/gcc-native/13/bin/g++
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9
export PATH="$CUDA_HOME/bin:/opt/cray/pe/gcc-native/13/bin:$HOME/.cargo/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=8.0
export MAX_JOBS=8 NVCC_THREADS=2
export PIP_CACHE_DIR=/lus/grand/projects/Intel/$USER/pip-cache
export TMPDIR=/lus/grand/projects/Intel/$USER/pip-tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

ENV_PATH="$(python -c 'import sys; print(sys.prefix)')"
NV="$ENV_PATH/lib/python3.12/site-packages/nvidia"
export CPATH="$NV/cublas/include:$NV/cudnn/include:$NV/nccl/include:\
$NV/cufft/include:$NV/curand/include:$NV/cusolver/include:\
$NV/cusparse/include:$NV/cusparselt/include:$NV/nvshmem/include:\
$NV/nvtx/include:$NV/cuda_cupti/include:$NV/cuda_nvrtc/include:\
$NV/cuda_runtime/include:${CPATH:-}"

cd ~/git/vllm
pip install -e . --no-build-isolation --no-deps -v   # ~85 min
exit
```

Back on the **login node**, finalize:

```bash
conda activate /lus/grand/projects/Intel/$USER/envs/monarch-cu128
grep -vE '^(torch|torchvision|torchaudio|torchcomms|triton)\b' \
    ~/git/vllm/requirements/common.txt > /tmp/vllm-runtime.txt
pip install -r /tmp/vllm-runtime.txt
pip install flashinfer-python --no-deps
python -c "import vllm; print('vllm:', vllm.__version__)"
```

### Alternative: editable install with pre-built .so files

If you already have the `.so` files built (from a prior `cmake --build` or
from copying them), you can register vLLM without rebuilding:

```bash
cd /lus/grand/projects/Intel/$USER/vllm
export TMPDIR=/lus/grand/projects/Intel/$USER/tmp
export PIP_CACHE_DIR=/lus/grand/projects/Intel/$USER/tmp/pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# VLLM_TARGET_DEVICE=empty skips all C++ extension builds
VLLM_TARGET_DEVICE=empty pip install -e . --no-build-isolation --no-deps
```

This creates a proper editable install using existing `.so` files in the source
tree. The `_qutlass_C` import warnings at runtime are benign (SM90+ only).

---

## 4. Download model and dataset

```bash
cd ~/git/torchtitan
python scripts/download_hf_assets.py \
    --repo_id Qwen/Qwen3-0.6B \
    --local_dir /lus/grand/projects/Intel/models/Qwen3-0.6B --all

python -c "from datasets import load_dataset; \
    load_dataset('kalomaze/alphabetic-arxiv-authors-it1')"
```

---

## 5. Run

Use an **interactive session** (batch jobs on preemptable get evicted quickly):

```bash
qsub -I -A Intel -q preemptable -l select=1:ncpus=64 \
    -l walltime=01:00:00 -l filesystems=home:grand

# On the compute node:
cd ~/git/torchtitan/torchtitan/experiments/rl
bash run_grpo_lora_a100.sh
```

Modes:

| MODE | Config | GPUs | LoRA | Notes |
|------|--------|------|------|-------|
| `grpo-lora-flex` (default) | `rl_grpo_qwen3_0_6b_flex_lora` | 4 | yes | flex attn trainer, FLASHINFER generator, cudagraph disabled |
| `flex` | `rl_grpo_qwen3_0_6b_flex` | 4 | no | |
| `grpo-lora` | `rl_grpo_qwen3_0_6b_varlen_lora` | 6 | yes | |
| `grpo` | `rl_grpo_qwen3_0_6b_varlen` | 6 | no | |

Architecture (default `grpo-lora-flex` mode on 40GB A100):
- **Trainer** (2 GPUs, TP2): flex attention, torch.compile per-block, LoRA rank=8
- **Generator** (2 GPUs, TP2): FLASHINFER attention backend, cudagraph disabled,
  `gpu_memory_limit=0.85`
- `TORCH_COMPILE_DISABLE=1` disables vLLM compile; trainer still uses per-block
  torch.compile (flex attention works through compile on A100)
- Generator always uses FLASHINFER regardless of model_spec attention type
  (flex crashes triton shared memory, varlen needs FA3 -- both broken on A100)

Overrides: `STEPS=20`, `SMOKE=1` (config-only, no GPU),
`CONDA_ENV_PATH=...`, `HF_ASSETS_PATH=...`.

Do **not** use `torchrun` -- Monarch manages GPU allocation internally.

---

## 6. Success criteria

- All 10 steps complete without error
- LoRA startup line: `LoRA training active with rank=8, alpha=16.0, target_modules=['wkv', 'wo', 'wq']`
- `reward_mean` increases pre -> post validation
- No CUDA OOM or NCCL timeout

---

## 7. Troubleshooting

### Build issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| vLLM build SIGKILL (exit 9) | Login node OOM | Build on compute node |
| `cublas_v2.h: No such file` | Missing dev headers | Set CPATH (see step 3) |
| `Errno 122: Disk quota exceeded` | Home 50 GB quota | Symlink vllm to grand; set `PIP_CACHE_DIR`+`TMPDIR` to grand |
| Build preempted mid-compile | Queue eviction | Use `cmake --build` directly for incremental resume |
| `Access to queue is denied` | Overdrawn allocation | Use `preemptable` queue |

### Dependency issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `torchstore requires torchmonarch==0.4.1` | Wrong version | `pip install torchmonarch==0.4.1` |
| `ModuleNotFoundError: flashinfer` | vLLM v1 requires it | `pip install flashinfer-python --no-deps` |
| flashinfer install downgrades torch | Has hard torch dep | Always `--no-deps`; reinstall torch if clobbered |
| flashinfer JIT: `cannot find -lcudart` or `curand.h not found` | Polaris HPC SDK missing libs/headers | Script creates merged `~/.local/flashinfer_cuda/` with symlinks (see script) |
| vLLM `Device string must not be empty` | Missing package metadata | Create dist-info (see shortcut install in step 3) |
| `ModuleNotFoundError: torchvision` | vLLM warmup imports it | `pip install torchvision==0.27.0.dev20260407+cu128 --extra-index-url ...nightly/cu128` |

### Attention backend issues (A100 + torch 2.12)

| Symptom | Cause | Fix |
|---------|-------|-----|
| Triton `OutOfMemoryError: out of resource` (332KB > 164KB) | torch.compile generates flex kernels exceeding A100 shared memory | Already fixed: cudagraph disabled + FLASHINFER backend |
| Flex attention OOM (12GB allocation in eager) | Without compile, flex uses math reference that materializes full attn matrix | Already fixed: switched to FLASHINFER backend |
| `num_splits requires FA3` | varlen/CUSTOM backend calls FA3 API unavailable on SM 8.0 | Already fixed: use `AttentionBackendEnum.FLASHINFER` instead of `CUSTOM` |

All three are resolved in the shipped config. The generator uses flashinfer
(pre-compiled CUDA kernels, no triton, no FA3, memory-efficient).

### Runtime issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: torchtitan` | PYTHONPATH not set | Run via `run_grpo_lora_a100.sh` (sets it automatically) |
| vLLM hangs at "Loading model weights" | TP mismatch | Shipped configs are safe (TP=2, TP=4) |
| OOM on trainer | Batch too large | Use LoRA mode, or reduce `local_batch_size` to 1 |
| OOM on generator (40GB A100) | gpu_memory_limit too high | Set `config.generator.gpu_memory_limit = 0.85` |
| LoRA "did not match any Linear.Config" | Wrong target names | Use `["wq", "wkv", "wo"]` for Qwen3 |
| LoRA "LMHeadCastConverter found no Linear" | Wrong converter order | LMHeadCastConverter must be before LoRAConverter |
| `nohup` job dies on SSH disconnect | Systemd reaps procs | Use `qsub` batch or `tmux` |
| Batch job killed in 2-3 min | Preemptable eviction | Use interactive `qsub -I` |

---

## 8. LoRA config details

```python
config.model_spec = model_registry(
    "0.6B", attn_backend="flex",
    converters=[
        LMHeadCastConverter.Config(),   # MUST be first
        LoRAConverter.Config(rank=8, alpha=16.0, target_modules=["wq", "wkv", "wo"]),
    ],
)
```

- `LMHeadCastConverter` before `LoRAConverter`: LoRA wraps non-targets
  in `FrozenConfig`, hiding `lm_head` from the cast converter.
- Qwen3 uses a single `wkv` Linear for both wk and wv,
  so target_modules is `["wq", "wkv", "wo"]`, not `["wq", "wk", "wv", "wo"]`.

---

## 9. Validated run results (2026-06-27)

### vLLM installation

vLLM installed via `VLLM_TARGET_DEVICE=empty pip install -e . --no-build-isolation --no-deps`
from pre-built source at `/lus/grand/projects/Intel/songhappy/vllm`.
Uses existing `.so` files; no cmake rebuild needed.

### Code changes (vs. upstream main)

1. **`torchtitan/experiments/rl/actors/generator.py`**: Hardcode
   `AttentionBackendEnum.FLASHINFER` for the vLLM generator, replacing the
   conditional flex/custom logic.
   - **Why**: On A100 (SM 8.0), flex attention crashes triton (332KB shared memory
     exceeds 164KB hardware limit), and varlen/CUSTOM requires FA3 which is
     unavailable on A100. FLASHINFER uses pre-compiled CUDA kernels that work.

2. **`torchtitan/experiments/rl/examples/alphabet_sort/config_registry.py`**: Add
   `rl_grpo_qwen3_0_6b_varlen_lora` and `rl_grpo_qwen3_0_6b_flex_lora` configs.
   - **Why**: Enable GRPO+LoRA on A100. Key decisions: LMHeadCastConverter must
     come before LoRAConverter (LoRA wraps non-targets in FrozenConfig, hiding
     lm_head from the cast converter). Qwen3 uses a single `wkv` Linear for both
     wk and wv, so target_modules is `["wq", "wkv", "wo"]`. gpu_memory_limit=0.85
     and cudagraph=disabled for 40GB A100 memory constraints.

3. **`torchtitan/models/common/decoder.py`**: Conditionally pass
   `separate_full_blocks` to `create_block_mask` by inspecting the function
   signature at runtime.
   - **Why**: torch 2.12.0.dev20260408 does not have this parameter. Without the
     check, the call crashes with an unexpected keyword argument error.

4. **`torchtitan/experiments/rl/trainer.py`**: Add `tyro.conf.Suppress` annotation
   to the `model_spec` field.
   - **Why**: `ModelSpec` is a complex nested object that cannot be parsed from CLI.
     Suppressing it prevents tyro from exposing it as a CLI flag (it is always set
     programmatically via config_registry).

5. **`torchtitan/experiments/rl/.claude/skills/inference_perf_hillclimb/SKILL.md`**:
   Add Polaris-specific operational notes (PBS vs nohup, OOM on login nodes,
   home quota management, compute node network isolation).
   - **Why**: Document hard-learned environment constraints so future work does
     not repeat failed approaches (e.g. nohup dying on SSH disconnect, nvcc
     OOM-killing login node builds).

### Training metrics (job 7222864, debug queue, 4x A100-40GB)

```
  Throughput Summary (2026-06-27, max_tokens=100, seq_len=2048)

  +--------------------------------------+-----------------+
  |                Metric                |      Value      |
  +--------------------------------------+-----------------+
  | Training (tokens/s)                  | 96 - 142        |
  +--------------------------------------+-----------------+
  | Inference throughput (tokens/s)      | 50 - 78         |
  +--------------------------------------+-----------------+
  | Step time (s)                        | 139 - 245       |
  +--------------------------------------+-----------------+
  | Reward mean                          | 0.15 - 0.38     |
  +--------------------------------------+-----------------+
  | Reward max (all steps)               | 1.0             |
  +--------------------------------------+-----------------+
```

| Step | Loss | Reward Mean | Tokens/s | Step Time |
|------|------|-------------|----------|-----------|
| 1 | -0.0018 | 0.38 | 96.4 | 239.6s |
| 2 | -0.0015 | 0.30 | 102.1 | 174.5s |
| 3 | -0.0032 | 0.33 | 129.3 | 161.4s |
| 4 | 0.00052 | 0.26 | 110.9 | 165.3s |
| 5 | -0.0028 | 0.15 | 142.3 | 139.2s |
| 6 | -0.0031 | 0.22 | 98.1 | 245.0s |

Note: Step 1 includes torch.compile warmup overhead.
First run hit 30-min walltime at step 7; resubmitted with 1h walltime (job 7222930).

### Key observations

- LoRA reduces optimizer memory by ~4x (rank=8, ~20MB trainable params)
- Generator KV cache: 606K tokens (32.38 GiB), 148x max concurrency
- Prefix cache hit rate: ~13-17% after warmup
- No OOM on 40GB A100s with `gpu_memory_limit=0.85`
- Weight sync (TorchStore) completes reliably between steps

---

## 10. Next steps (week of 2026-06-30)

1. **Complete full 10-step validation** with post-training validation showing
   reward improvement (job 7222930 running with 1h walltime)
2. **Re-enable torch.compile** once torch nightly adds `separate_full_blocks`
   support, or pin a newer nightly that includes it
3. **Upgrade to newer torch nightly** (post-April 2026) that fixes FA3 on SM 8.0,
   which would allow varlen attention for both trainer and generator
4. **Enable cudagraph** for generator once triton flex attention shared memory
   issue is resolved (significant inference speedup expected)
5. **Run with larger models** (Qwen3-1.7B) to stress-test memory limits on 40GB A100
6. **Multi-node validation** (2+ nodes) requiring HostMesh Monarch configuration
7. **Performance optimization**: profile the ~245s step time variance (steps 1/6
   are 2x slower than step 5; likely generation variability)
