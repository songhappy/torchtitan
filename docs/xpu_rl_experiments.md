# GRPO on Intel XPU: experiments and results

What we ran on Aurora and what came out, for the [roadmap](xpu_rl_roadmap.md). How to build and run the stack, and
how to judge a run: [xpu_torch215.md](xpu_torch215.md). Pass criteria: [A healthy run](xpu_torch215.md#a-healthy-run);
open problems: [Known issues](xpu_torch215.md#known-issues).

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

Result: 16/16. `TRN_COPIES=2` over a subset of nodes (2 x 48) hangs -- [known](xpu_torch215.md#known-issues).

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
([known issue](xpu_torch215.md#known-issues)).

### 8. Logprob parity (trainer vs generator)

Every run prints `bit_wise/` metrics. Mean log-ratio: Qwen3-0.6B ~-0.0007, Qwen3-30B-A3B ~-0.01 (~7% of tokens
differ). The max exceeds 10 on the larger models ([known issue](xpu_torch215.md#known-issues)).
