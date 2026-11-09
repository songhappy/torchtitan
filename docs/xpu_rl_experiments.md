# GRPO on Intel XPU: experiments and results

Every experiment the regression runs, with its runs, meshes and latest result. Updated at milestones; every run's
full results are in its generated regression report. Build and validate: [xpu_torch215.md](xpu_torch215.md) (it lists
the rows below that pass on the current env, with their `qsub` lines); gaps: [xpu_rl_roadmap.md](xpu_rl_roadmap.md).

Model Qwen3-0.6B (alphabet sort, both arms: each row is a full and a LoRA run) unless noted. Trainer mesh =
`dp_replicate x dp_shard` (FSDP) unless it says TP; generator mesh = vLLM engines x tiles per engine (tiles per
engine = the engine's TP). Each section is one `sweep.sh` step of the regression tooling (`pr` = steps 1, 2, 5, 7;
`milestone` = all seven). Stacks: **2.15.0** = torch 2.15.0 + Monarch main (current, 11-09); **nightly** = the
previous torch 2.15 nightly stack (same vLLM, TorchStore, oneCCL).

### 1. Smoke (after every stack change)

`step1`, 4 runs, 2 steps: the loop on one host (shared memory) and across nodes (XCCL).

| Runs | Steps | Trainer mesh | Generator mesh | Latest result |
|---|---|---|---|---|
| 1 node: 2 trainer + 2 generator tiles | 2 | 1 x 2 | 2 x 1 (same node) | pass (11-09, 2.15.0) |
| 8 nodes x 4 tiles | 2 | 4 x 4 (nodes 1-4) | 16 x 1 (nodes 5-8) | not run yet (the 2-node layout it replaced passed 11-09) |

### 2. Basic regression (18 runs; the gate for any code change)

`step2`, 18 runs: every tile count on one trainer node, 16-node scale-out, and 10-step runs.

| Runs | Steps | Trainer mesh | Generator mesh | Latest result |
|---|---|---|---|---|
| 2 nodes x P tiles, P = 1, 2, 4, 8, 12 | 2 | 1 x P (node 1) | P x 1 (node 2) | pass, 10/10 runs (11-09, 2.15.0) |
| 16 nodes x P, P = 4, 12 | 2 | 8 x P (nodes 1-8) | 8P x 1 (32 / 96; nodes 9-16) | pass, 4/4 (11-09, 2.15.0); peak 2134 of 8192 tasks, 128 GB host per node |
| 1 node, 10 steps | 10 | 1 x 2 | 2 x 1 (same node) | pass, 2/2 (11-09, 2.15.0) |
| 8 nodes x 4, 10 steps | 10 | 4 x 4 (nodes 1-4) | 16 x 1 (nodes 5-8) | pass, 2/2 (11-09, 2.15.0) |

### 3. Weight transport at scale

`step3`, 6 runs, full arm only (every weight moves), 2 steps: every way TorchStore moves weights.

| Runs | Steps | Trainer mesh | Generator mesh | Latest result |
|---|---|---|---|---|
| 1 node x 4 (shared memory) | 2 | 1 x 2 | 2 x 1 (same node) | pass, 6/6 for the step (11-02, nightly) |
| 2 nodes x 4 (XCCL) | 2 | 1 x 4 (node 1) | 4 x 1 (node 2) | |
| 2 nodes x 4, generators TP 2 (resharding into TP shards) | 2 | 1 x 4 (node 1) | 2 x 2 (node 2) | |
| 16 nodes x 4 (32 pullers of 8 copies) | 2 | 8 x 4 (nodes 1-8) | 32 x 1 (nodes 9-16) | |
| 16 nodes x 12 (96 pullers of 8 copies) | 2 | 8 x 12 (nodes 1-8) | 96 x 1 (nodes 9-16) | |
| 16 nodes x 12, one copy (96 communicators per volume; needs TorchStore's routing fix) | 2 | 1 x 96 (nodes 1-8) | 96 x 1 (nodes 9-16) | |

### 4. Trainer scaling, one copy (generators: 12 single-tile engines on the last node)

`step4`, 18 runs, 2 steps: one trainer copy, first on more tiles of one node, then over more nodes.

| Runs | Steps | Trainer mesh | Generator mesh | Latest result |
|---|---|---|---|---|
| 2 nodes x 12, trainer on A tiles, A = 1, 2, 4, 8, 12 | 2 | 1 x A (node 1) | 12 x 1 (node 2) | pass, 16/16 for the scaling rows (10-29, nightly) |
| G + 1 nodes x 12, one copy over G nodes, G = 2, 4, 8 | 2 | 1 x 12G (24 / 48 / 96; nodes 1-G) | 12 x 1 (last node) | |
| 16 nodes x 12, two copies across nodes (HSDP) | 2 | 2 x 48 (nodes 1-8) | 96 x 1 (nodes 9-16) | hangs (10-29, nightly; [known issue](xpu_torch215.md#known-issues)) |

### 5. Generator scaling, one engine (trainer fixed)

`step5`, 12 runs, 2 steps: one vLLM engine grows by tensor parallelism; TP must divide the 8 KV heads, so an engine
spans nodes only on 4-tile nodes.

| Runs | Steps | Trainer mesh | Generator mesh | Latest result |
|---|---|---|---|---|
| 2 nodes x 12, engine TP A, A = 1, 2, 4, 8 | 2 | 1 x 12 (node 1) | 1 x A (node 2) | pass, 12/12 for the step (10-29, nightly); re-run on 2.15.0 in progress |
| 3 nodes x 4, one engine over 2 nodes | 2 | 1 x 4 (node 1) | 1 x 8 (nodes 2-3) | |
| 9 nodes x 4, four such engines | 2 | 1 x 4 (node 1) | 4 x 8 (nodes 2-9) | |

### 6. Both sides at scale

`step6`, 16 runs: the default layout on full nodes, both sides growing together; 2 steps for correctness, 10 steps
with the batch sized to the trainer for throughput.

| Runs | Steps | Trainer mesh | Generator mesh | Latest result |
|---|---|---|---|---|
| N nodes x 12, N = 2, 4, 8, 16 | 2 | N/2 x 12 (first half) | 6N x 1 (12 / 24 / 48 / 96; second half) | pass, 12/12 for the step (10-29, nightly) |
| N nodes x 12, batch sized to the trainer, N = 2, 4, 8, 16 | 10 | N/2 x 12 | 6N x 1 | |

### 7. Larger models (flex attention; `CONFIG=...`, 10 steps)

`step7`, 6 runs (the first six rows): `_flex` = upstream's recipe with flex attention (varlen has no XPU kernel);
`_16n_...` = our 16-node layouts (FSDP instead of trainer TP / EP, no checkpoint save).

| Runs | Steps | Trainer mesh | Generator mesh | Latest result |
|---|---|---|---|---|
| Qwen3-14B, upstream `rl_grpo_qwen3_14b_flex`, 2 nodes | 10 | TP 8 (node 1) | 1 x 8 (node 2) | pass (11-09, 2.15.0) |
| Qwen3-1.7B `rl_grpo_qwen3_1_7b_flex_16n_ppn8_gen_c8_10steps`, 16 nodes | 10 | 8 x 8 (nodes 1-8) | 8 x 8 (nodes 9-16) | pass (11-09, 2.15.0) |
| Qwen3-1.7B, upstream `rl_grpo_qwen3_1_7b_flex`, 1 node | 10 | TP 2 | 1 x 4 (same node) | 10/10 but fails the parity check, max 13.3 (11-09, 2.15.0) |
| Qwen3-30B-A3B `..._16n_ppn8_trn_c1_gen_c8_10steps`, 16 nodes | 10 | 1 x 64 (nodes 1-8) | 8 x (TP 4 x DP 2, EP 8), vLLM at 0.7 (nodes 9-16) | 10/10 but fails the parity check, max 24.4 (11-09, 2.15.0) |
| Qwen3-14B `..._16n_ppn8_gen_c8_10steps`, 16 nodes, full activation remat | 10 | 8 x 8 (nodes 1-8) | 8 x 8 (nodes 9-16) | still training at walltime, ~4.5 min/step (10-29, nightly); re-run on 2.15.0 in progress |
| gpt-oss-20b `..._16n_ppn4_trn_c1_gen_c8_10steps`, 16 nodes | 10 | 1 x 32 (nodes 1-8 x 4) | 8 x (TP 4, EP 4), vLLM at 0.7 (nodes 9-16) | still training at walltime (10-29, nightly); re-run on 2.15.0 in progress |
| Qwen3-30B-A3B, upstream, 1 node (not in step 7) | 10 | FSDP 2 x TP 2 x EP 4 | DP 2 x TP 2 x EP 4 | fails at generator start: oneCCL KVS errors (10-29, nightly) |
| gpt-oss-20b, upstream, 1 node (not in step 7) | 10 | TP 2 x EP 2 (2 tiles) | TP 4 x EP 4 | trainer out of memory (10-29, nightly) |
| Qwen3.5-9B / Qwen3.6-27B (not in step 7) | -- | -- | -- | blocked: vLLM's GDN path needs NVIDIA-only kernels (10-29, nightly) |

Why these layouts: one MoE copy per 8-tile node runs out of memory at step 2; vLLM at 0.9 of a tile fails the first
forward; FSDP shards must divide the FFN width and the expert count; TP must divide the KV heads.

### 8. Logprob parity (trainer vs generator)

`bit_wise/` metrics in every run. Mean log-ratio: Qwen3-0.6B ~-0.0007, Qwen3-30B-A3B ~-0.01 (~7% of tokens differ);
the max exceeds 10 on the larger models (open; [known issue](xpu_torch215.md#known-issues)).
