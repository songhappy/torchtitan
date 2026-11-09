# GRPO on Intel XPU: experiments and results

One entry per experiment: what it tests, how to run it, the latest result. Updated at milestones only; every run's
full results are in its generated regression report. Build and validate: [xpu_torch215.md](xpu_torch215.md); gaps:
[xpu_rl_roadmap.md](xpu_rl_roadmap.md). Model Qwen3-0.6B (alphabet sort, both arms) unless noted; mesh = trainer
`dp_replicate x dp_shard`; "group" = `sweep.sh` step of the regression tooling (`pr` = steps 1, 2, 7; `milestone` = all seven); env = `monarch_t215`.

### 1. Smoke (after every stack change)

The loop on one host (shared memory) and across nodes (XCCL), 1n / 8n x 4. Group `step1` (4). Latest: 4/4 with the
earlier 2n x 4 layout (11-09, torch 2.15.0 + Monarch main); 8n x 4 not run yet.

### 2. Basic regression (18 runs; the gate for any code change)

Default layout at 2n x 1-12 tiles, 16n x 4 / 12, and 10 steps at 1n / 8n. Group `step2` (18). Latest: 18/18, peak
2134 of 8192 tasks and 128 GB host memory per node (11-09, torch 2.15.0 + Monarch main).

### 3. Weight transport at scale

TorchStore over XCCL to 96 pullers: 8 x 4, 8 x 12, 1 x 96 (needs TorchStore's routing fix). Group `step3` (6).
Latest: 6/6 (11-02, torch 2.15 nightly).

### 4. Trainer scaling, one copy (generators: 12 single-tile engines on the last node)

One copy on 1-12 tiles, then 1 x 24 / 48 / 96. Group `step4` (18). Latest: 16/16; two copies spanning nodes (HSDP)
hang (10-29, nightly).

### 5. Generator scaling, one engine (trainer fixed)

One engine of TP 1-8, one TP-8 engine over 2 nodes, four such on 8 nodes. Group `step5` (12). Latest: 12/12 (10-29,
nightly).

### 6. Both sides at scale

Default layout at 4n / 8n x 12, and the batch sized to the trainer on 2-16 nodes. Group `step6` (16). Latest: 12/12
(10-29, nightly).

### 7. Larger models (flex attention; `CONFIG=...`, 10 steps)

Group `step7` (6). `_flex` = upstream's recipe with flex attention (varlen has no XPU kernel); `_16n_...` = our 16-node layouts (FSDP
instead of trainer TP / EP, no checkpoint save).

| Model | Layout | Result | Date, stack |
|---|---|---|---|
| Qwen3-1.7B | upstream: trainer TP 2, one TP-4 engine (1 node) | 10/10; parity max 13.3 | 11-09, torch 2.15.0 |
| Qwen3-14B | upstream: trainer TP 8, one TP-8 engine (2 nodes) | 10/10 | 11-09, torch 2.15.0 |
| Qwen3-30B-A3B | upstream: FSDP 2 x TP 2 x EP 4 (1 node) | fails at generator start (oneCCL KVS errors) | 10-29, nightly |
| gpt-oss-20b | upstream: trainer TP 2 x EP 2 on 2 tiles | trainer out of memory | 10-29, nightly |
| Qwen3.5-9B / 3.6-27B | any | blocked: vLLM's GDN path needs NVIDIA-only kernels | 10-29, nightly |
| Qwen3-1.7B | 16n: 8 x 8 FSDP, 8 TP-8 engines | 10/10 | 11-09, torch 2.15.0 |
| Qwen3-14B | 16n: 8 x 8 FSDP + full remat, 8 TP-8 engines | still training at walltime (~4.5 min/step) | 10-29, nightly |
| Qwen3-30B-A3B | 16n: 1 x 64 FSDP, 8 TP4 x DP2 / EP8 engines, vLLM 0.7 | 10/10; parity max 24.4 | 11-09, torch 2.15.0 |
| gpt-oss-20b | 16n: 1 x 32 FSDP, 8 TP4 / EP4 engines, vLLM 0.7 | still training at walltime | 10-29, nightly |

Why these layouts: one MoE copy per 8-tile node runs out of memory at step 2; vLLM at 0.9 of a tile fails the first
forward; FSDP shards must divide the FFN width and the expert count; TP must divide the KV heads.

### 8. Logprob parity (trainer vs generator)

`bit_wise/` metrics in every run. Mean log-ratio: Qwen3-0.6B ~-0.0007, Qwen3-30B-A3B ~-0.01 (~7% of tokens differ);
the max exceeds 10 on the larger models (open; [known issue](xpu_torch215.md#known-issues)).
