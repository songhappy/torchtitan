# Perf Baseline: GRPO + LoRA (Flex Attention) on A100

**Date**: 2026-06-29
**Job**: 7229038 (Polaris debug queue)
**Hardware**: 4x A100-40GB, single node
**Commit**: main (739bb9a9) + uncommitted LoRA/flex/decoder patches

---

## Configuration

### Model
- Architecture: Qwen3-0.6B (28 layers, dim=1024, 16 heads, 8 KV heads, head_dim=128)
- Vocab: 151,936; max_seq_len: 4096; weight tying: enabled
- RoPE: theta=1,000,000

### LoRA
- Rank: 8, Alpha: 16.0
- Targets: [wq, wkv, wo]
- Converters: LMHeadCastConverter (fp32 lm_head) -> LoRAConverter

### Trainer (2 GPUs, TP=2)
- Attention: flex (triton)
- Compile: enabled, backend=aot_eager, components=[model, loss]
- Dtype: bfloat16
- Parallelism: FSDP (dp_shard=1) + TP=2
- AC: SelectiveAC (default)
- Optimizer: AdamW (lr=2e-6, betas=(0.9, 0.95), eps=1e-8, wd=0.1)
- LR schedule: linear decay, warmup=2 steps
- Loss: GRPO (clip_eps=0.2)

### Generator (2 GPUs, TP=2)
- Engine: vLLM + FLASHINFER backend
- Cudagraph: DISABLED
- Compile (vLLM): DISABLED (TORCH_COMPILE_DISABLE=1)
- Dtype: bfloat16
- gpu_memory_limit: 0.85
- KV cache: 606,208 tokens (32.38 GiB), 148x concurrency
- block_size: 256
- Scheduling: FCFS
- max_engine_steps_between_decisions: 16

### Sampling
- Temperature: 0.8, top_p: 0.95, max_tokens: 100

### Batching / RL
- local_batch_size: 2, global_batch_size: 8, seq_len: 2048
- group_size: 8 (GRPO siblings per prompt)
- num_groups_per_rollout_batch: 5
- num_validation_samples: 20
- num_steps: 10
- Task: alphabet sort (alphabetic-arxiv-authors)
- Renderer: qwen3, thinking=disabled

### Software
- Torch: 2.12.0.dev20260408+cu128
- Triton: 3.5.1
- vLLM: source build (flashinfer backend)
- Monarch: torchmonarch 0.4.1

---

## Results

### Per-step metrics

| Step | tokens/s | step_time (s) | reward_mean | reward_max | loss | grad_norm | resp_len_max |
|------|----------|---------------|-------------|------------|------|-----------|--------------|
| 1 | 98.4 | 235.2 | 0.38 | 1.0 | -0.0024 | 0.041 | 62 |
| 2 | 100.0 | 178.2 | 0.30 | 1.0 | 0.00021 | 0.090 | 55 |
| 3 | 123.2 | 168.5 | 0.32 | 1.0 | -0.0026 | 0.046 | 57 |
| 4 | 107.0 | 171.8 | 0.30 | 1.0 | -0.0026 | 0.084 | 61 |
| 5 | 145.2 | 136.2 | 0.15 | 1.0 | -0.00080 | 0.063 | 69 |
| 6 | 101.5 | 236.0 | 0.23 | 1.0 | -0.0012 | 0.050 | 54 |
| 7 | 119.9 | 217.0 | 0.30 | 1.0 | -0.0050 | 0.10 | 60 |
| 8 | 113.7 | 185.4 | 0.17 | 1.0 | -0.0015 | 0.031 | 73 |
| 9 | 101.8 | 181.0 | 0.19 | 0.86 | -0.0093 | 0.096 | 57 |
| 10 | 106.3 | 166.9 | 0.21 | 1.0 | -0.00037 | 0.034 | 64 |

### Aggregate performance

| Metric | Value |
|--------|-------|
| **Avg tokens/s (steps 2-10)** | **113.2** |
| **Avg step time (steps 2-10)** | **182.8s** |
| Avg tokens/s (all steps) | 111.7 |
| Avg step time (all steps) | 187.6s |
| Step time range | 136.2 - 236.0s |
| Step time stddev (steps 2-10) | 28.6s |
| Total training wall time | ~31 min (steps only) |
| Validation time (pre) | 90.25s |
| Validation time (post) | 70.59s |

### Validation reward (pre/post training)

| Metric | Pre | Post | Delta |
|--------|-----|------|-------|
| reward_mean | +0.181 | +0.314 | +0.133 (+73%) |
| reward_max | +0.514 | +1.000 | +0.486 |
| reward_min | +0.000 | +0.012 | +0.012 |
| reward_std | +0.145 | +0.258 | +0.113 |

### Generator throughput (from vLLM 10s stats, peaks)

- Prompt throughput: 108-522 tokens/s
- Generation throughput: 35-85 tokens/s
- Prefix cache hit rate: ~16% at end

---

## Bottleneck analysis

1. **Step time variance (136-236s)**: driven by generation length variability and
   row-dropping (packing to global_batch_size=8). Steps with more diverse/longer
   rollouts take longer in the generation phase.

2. **Low generator utilization**: cudagraph disabled, compile disabled, FLASHINFER
   only. Enabling cudagraph (FULL_DECODE_ONLY mode) would accelerate decode.

3. **aot_eager compile**: trainer uses aot_eager (no inductor optimizations).
   Switching to inductor backend would speed up fwd/bwd but requires fixing
   the separate_full_blocks compat issue first.

---

## Optimization opportunities (for future comparison)

1. Enable cudagraph (FULL_DECODE_ONLY) for generator
2. Switch trainer compile to inductor backend
3. Upgrade torch nightly (fix separate_full_blocks, FA3 on SM 8.0)
4. Enable vLLM compile (VLLM_TORCH_COMPILE_LEVEL > 0)
5. Increase max_tokens (longer rollouts may improve reward signal)
6. Profile generation vs training time split per step
