# XPU RL roadmap: GRPO on Intel XPU vs upstream CUDA

What it takes to run upstream torchtitan RL (async GRPO: Monarch actors, TorchStore weight sync, vLLM generation,
FSDP2 training) on Intel XPU, and how far each piece is. This file is the gap list; how to build, run and validate
the stack is [xpu_torch215.md](xpu_torch215.md). Updated 2026-11-09; update after every step (set the status, add the
date). Start point: upstream torchtitan RL ran on CUDA only, and nothing in the stack worked on XPU.

Contents: [Phases](#phases) - [Phase 0](#phase-0----each-component-on-xpu) -
[Phase 1](#phase-1----qwen3-06b-grpo-end-to-end) - [Phase 2](#phase-2----bigger-models-more-parallelism) -
[Phase 3](#phase-3----rdma-on-three-intel-platforms-cxi-aurora-bmg-and-crescent-island) - [Platform support](#platform-support) -
[Notes](#notes)

## Phases

| Phase | Goal | Model |
|---|---|---|
| **0** | each component works on its own on XPU | Qwen3-0.6B where a model is needed |
| **1** | small-model GRPO end to end, 1-16 nodes | Qwen3-0.6B, full + LoRA |
| **2** | bigger models, more parallelism on trainer and generator | Qwen3-1.7B .. 30B-A3B, gpt-oss-20b |
| **3** | RDMA: one-sided weight and rollout transfer; tracks 3A CXI (Aurora: PVC + Slingshot), 3B BMG (Arc Battlemage) and 3C Crescent Island | Qwen3-0.6B, then 30B-A3B |

Within a phase rows go stack (S), Monarch (M), TorchStore (TS), torchtitan (T), vLLM (V), in the order they are done;
IDs are `<phase>-<repo><n>`. Column **Upstream** = what has to land upstream (PR or issue and its state; "--" when
nothing does).

**Status:** `done` validated on Aurora - `workaround` works with a workaround - `review` done, PR in review or local -
`FAILS` fails today - `todo` not done. Dates are when the status was reached (2026).

**Summary: 93 items -- 27 done, 14 workaround, 16 review, 3 FAILS, 33 todo.** Phases 0-1 are done apart from reviews:
GRPO (Qwen3-0.6B, full + LoRA) runs on Aurora up to 16 nodes x 12 tiles; both Monarch PRs are merged; TorchStore #171 /
#201 / #221 and torchtitan #5402-#5404 are in review. Phase 2 runs the larger models with workarounds (HSDP hangs);
phase 3A (CXI) trains GRPO with NIXL weights and rollouts on 2-16 nodes; phase 3B (BMG) has not started.

## Phase 0 -- each component on XPU

### Stack (PyTorch, torch-xpu-ops, oneAPI)

| # | Item | XPU now | Status | Upstream |
|---|---|---|---|---|
| 0&#8209;S1 | PyTorch with XPU | torch 2.15.0+xpu (release; was 2.14.0 / 2.15 nightly), triton-xpu 3.9.0 | done 11-09 | -- |
| 0&#8209;S2 | oneAPI / oneCCL | oneAPI 2026.1, oneCCL 2022.1.2; env `monarch_t215` = torch 2.15.0 + Monarch upstream main (rebuilt 11-09: compute checks, smoke 4/4, basic 18/18); `monarch_t214` (torch 2.14) kept as the old baseline | done 11-09 | -- |
| 0&#8209;S3 | RMSNorm backward, large batches | stock kernel fixed | done 09-14 | torch-xpu-ops #4790 (merged) |
| 0&#8209;S4 | oneCCL bugs with C reproducers (#222 broadcast scale-out, #223 worker affinity, #225 concurrent communicator creation) | workarounds in place (0-T4, 1-T6, 2-V1) | workaround | oneCCL #222-#225 filed, no fixes yet |
| 0&#8209;S5 | A process that used an XCCL group segfaults at exit (one rank, after all checks pass) | 24/24 runs, torch 2.14 + 2.15; clean without a process group; harmless for GRPO, fails 2-node test launchers | FAILS 11-04 | to file: torch-xpu-ops `ProcessGroupXCCL` or oneCCL |
| 0&#8209;S6 | Same torch / vLLM as upstream main (CUDA uses the nightlies) | vLLM main pins torch 2.15.0 for XPU, but drops the PassConfig sequence-parallel fields torchtitan's generator passes (#60517), so we stay on vLLM `083060d04` | todo | torchtitan generator for post-#60517 vLLM |

### Monarch

| # | Item | XPU now | Status | Upstream |
|---|---|---|---|---|
| 0&#8209;M1 | Build without the tensor engine (`USE_TENSOR_ENGINE=0`, actors only) | builds | done 10-05 | -- |
| 0&#8209;M2 | Actors on XPU devices (upstream counted CUDA devices) | XPU device count, XPU env vars in proc meshes | done 11-03 | #4307 merged (8f5f5198b) |
| 0&#8209;M3 | PBS clusters: `PBSJob` (upstream had `SlurmJob` only) | multi-node host meshes on PBS; e2e PASS on Aurora 2 x 12 (XCCL) and Polaris 2 x 4 A100 (NCCL) | done 10-29 | #5067 merged (0daf3d332) |

### TorchStore

| # | Item | XPU now | Status | Upstream |
|---|---|---|---|---|
| 0&#8209;TS1 | TorchStore build on XPU | XPU branch `xpu-all` on upstream main `e500a6f` | done 10-05 | -- |
| 0&#8209;TS2 | Rebase onto upstream main | `xpu-all` on `e500a6f`; XCCL appended to `TransportType` (enum values kept), `gloo.py` left as upstream's, pickling fix | done 11-03 | -- |
| 0&#8209;TS3 | Host name from `TORCHSTORE_HOSTNAME` (#226), replacing `unset HOSTNAME` | not adopted in the launcher | todo | -- (upstream has it) |
| 0&#8209;TS4 | SharedMemory (same host) made XPU-safe | in #171 | review | #171 (open, see 0-TS5) |
| 0&#8209;TS5 | XCCL transport: GPU to GPU over Slingshot (upstream CUDA uses RDMA) | #171 rebased + trimmed (`3d0b0b3`): XPU tests 6/6, CI CPU suite pass; #221 (`d149d24`) = #171 + 2-node tests, oneCCL #222 reproducer xfail without the workaround | review 11-03 | #171 (open: waits on amirafzali, CHANGES_REQUESTED 07-13, re-requested; check fit with the `_transport` transport #232 and #255); #221 (draft: #171 + 2-node tests); both wait for workflow approval |
| 0&#8209;TS6 | XCCL handshake port race (`EADDRINUSE`) | bind port 0, with a test | review 11-03 | in #171 (open); held locally until #171 moves |
| 0&#8209;TS7 | Gloo (CPU staged), MonarchRPC (last resort) | work; speed vs XCCL measured in phase 3 (XCCL ~4x Gloo) | done | -- |

### torchtitan

| # | Item | XPU now | Status | Upstream |
|---|---|---|---|---|
| 0&#8209;T1 | RFC: torchtitan RL on XPU (scope, plan, PR split) | posted 10-15 | review 10-15 | pytorch/torchtitan#5401 |
| 0&#8209;T2 | Aurora launcher: PALS `mpiexec` by path, `--cpu-bind none`, `TMPDIR`, `unset HOSTNAME` | `run_grpo.sh` | done | -- (site script) |
| 0&#8209;T3 | Network: CXI / oneCCL env block, newest libfabric, 256 MiB host buffer | launcher | done | -- (site script) |
| 0&#8209;T4 | oneCCL 2022: `CCL_OP_SYNC`, `CCL_WORKER_AFFINITY` unset | launcher | workaround 10-06 | oneCCL #223 |
| 0&#8209;T5 | Device visibility of spawned procs (upstream: `CUDA_VISIBLE_DEVICES`) | `ZE_AFFINITY_MASK` from `device_type` | review 10-15 | #5402 (draft; conflicts with main: rebase first, 1-T17) |
| 0&#8209;T6 | Flex attention, compiled (varlen has no XPU kernel) | same as upstream | done 10-14 | -- |
| 0&#8209;T7 | Load HF checkpoint | pass | done | -- |
| 0&#8209;T8 | CPU unit tests for the XPU paths | in the PRs | review | with #5402-#5404 |
| 0&#8209;T9 | Minimal tests per code change | `pr_test` skill: lint, recipe-grid check, unit tests, library checks | review 10-27 | -- (our tooling) |

### vLLM

| # | Item | XPU now | Status | Upstream |
|---|---|---|---|---|
| 0&#8209;V1 | vLLM `main` built for XPU + `vllm-xpu-kernels` (no vLLM changes needed) | generates ("Paris") | done 10-05 | -- |
| 0&#8209;V2 | Generation correctness: TP 1 / 2 / 4 x flash attention / Triton, same first tokens | 6/6 PASS; TP > 1 dumps core at exit | review 10-26 | -- |

## Phase 1 -- Qwen3-0.6B GRPO end to end

### Monarch

| # | Item | XPU now | Status | Upstream |
|---|---|---|---|---|
| 1&#8209;M1 | Rollouts through the controller, router and actors; 96-actor fan-out | 6/6 | done | -- |
| 1&#8209;M2 | Actor messages (rollouts, `generate()`, replies) on the fast network: `HYPERACTOR_TCP_INTERFACE=hsn0` binds and advertises that interface (upstream binds `gethostname()`, on Aurora the ~1 Gb management network, ~0.1 GB/s) | patch + tests local (`~/git/monarch-m1` d868f9a3e, 81 lines, 8/8 channel tests); 2-7x per message; no end-to-end change at 0.6B (~1 MB / step) | workaround 11-06 | Monarch PR: patch local, not pushed |

### TorchStore

| # | Item | XPU now | Status | Upstream |
|---|---|---|---|---|
| 1&#8209;TS1 | Default client: storage volumes (2 hops), controller lookup on every fetch | used | done | -- |
| 1&#8209;TS2 | Routing client: per-rank routes planned once (#183-#224), used by torchtitan main | used since the 10-28 rebase; with XCCL each rank reads one replica (`Balance.REQUESTER`, branch `routing-requester-balance`) | review | requester balance: PR draft |
| 1&#8209;TS3 | Fetch each replicated region once (upstream TODO since #116) | dedup; CI unit suite passes; pushed `4553f7f` | review 11-03 | #201 (open, no reviewer) |
| 1&#8209;TS4 | XCCL limit: one communicator per pair, oneCCL max 128 per process (CUDA: RDMA, no communicators) | `1 x 96` fails; fix = reuse communicators (one per host pair, or one per generator); phase 3 removes the cap | FAILS | in #171 if fixed |

### torchtitan

| # | Item | XPU now | Status | Upstream |
|---|---|---|---|---|
| 1&#8209;T1 | Qwen3-0.6B | pass | done 10-05 | -- |
| 1&#8209;T2 | GRPO on alphabet_sort | 200 steps, reward rising, 1 and 8 nodes | done 10-15 | -- |
| 1&#8209;T3 | LoRA | pass | done 10-14 | -- |
| 1&#8209;T4 | Multi-host driver (upstream: `this_host()`; 11-03 `launcher/` package, SLURM first, #5572) | `TORCHTITAN_MONARCH_JOB` (Slurm or PBS); rework as a PBS launcher on `launcher/base.py` | review 10-15 | #5403 (draft; overlaps the upstream launchers #5572: rework as a PBS launcher) |
| 1&#8209;T5 | Scale: thread caps (8192-task cgroup), node-local inductor cache, Level Zero v2, `FLAT` tiles | launcher | done 10-14 | -- (site script) |
| 1&#8209;T6 | oneCCL #222 / #225 workarounds | `storage_volumes_in_own_procs`, `sequential_initial_pull`, on for XPU | workaround 10-15 | #5404 (draft, clean) |
| 1&#8209;T7 | Allocator safe for transfers (upstream: `expandable_segments`) | not set on XPU (corrupts oneCCL transfers) | review 10-15 | #5402 (draft) |
| 1&#8209;T8 | FSDP2, one node | pass | done 10-14 | -- |
| 1&#8209;T9 | FSDP2, one copy per node | up to 16 x 12 | done 10-14 | -- |
| 1&#8209;T10 | FSDP2, one copy across nodes | `1 x 8 / 16 / 32` pass | done 10-16 | -- |
| 1&#8209;T11 | FSDP2, one copy over 96 tiles | pass on torch 2.15 (`trn_n8`, both arms, 10-29); intermittent code 11 on torch 2.14 (1-TS4) | workaround 10-29 | -- |
| 1&#8209;T12 | Recipes for nodes x tiles x layouts | `alphabet_sort.py` grid (full / LoRA; `trn_c`, `gen_tp`, `trn_t`, `gen_t`, `batch_scaled`; local `847599960`) | done 10-14 | -- (local recipes) |
| 1&#8209;T13 | Regression suite | `xpu_grpo_regression`: steps 1-7; `pr` (steps 1, 2, 5, 7) per PR, `milestone` (all seven, published report) per milestone | done 11-09 | -- (our tooling) |
| 1&#8209;T14 | Scale: 1-16 nodes, 1-12 tiles per node | 25/31; failures = 1-TS4, 2-T2 | done 10-16 | -- |
| 1&#8209;T15 | Save checkpoint | off (`load_only`); OOM in oneCCL at 8n on torch 2.12 | todo | -- |
| 1&#8209;T16 | Prompt data state in checkpoints (#5114) | not on our base | todo | -- (adopt on rebase) |
| 1&#8209;T17 | Rebase: role packages (#5316), generator split (#5317), `ActorSystem` (#5391) | rebased onto main 10-28 (`rl-xpu-aurora-torch215`); main moved again (117 commits by 11-06) | workaround | needed before #5402-#5404 land |
| 1&#8209;T18 | `torch.accelerator` for device / stream / memory (#5450); may shrink 0-T5, 1-T7 | -- | todo | -- (adopt on rebase) |
| 1&#8209;T19 | Weights staged in pinned CPU memory (#4945, #5107); upstream dropped the trainer copy again on 11-06 (generator `cpu_prefetch` / `gpu_fetch`) | not on our base | todo | -- (adopt on rebase) |
| 1&#8209;T20 | Layout parity: upstream TP 2 + TP 2 vs ours, 200 seeded steps | configs ready (local `ed952764d`); not run | todo | -- |
| 1&#8209;T21 | Scaling: both sides 2-16n x 12, batch sized to the trainer | configs ready; not run | todo | -- |
| 1&#8209;T22 | Soak: 8n x 12, 1000 steps | config ready; not run | todo | -- |
| 1&#8209;T23 | Training throughput vs torch 2.12: full 8-node `fwd_bwd` ~30% slower (22063 vs 32411 tok/s) | not investigated | todo | -- |

### vLLM (generator in the loop)

| # | Item | XPU now | Status | Upstream |
|---|---|---|---|---|
| 1&#8209;V1 | vLLM in the RL loop: eager, XPU attention backend, `XPUWorker` subclass (upstream: CUDA graphs, CUDA `Worker`; still CUDA-only in torchtitan main: `TorchTitanGPUWorker`, `torch.cuda` calls) | pass, up to 96 generators | review 10-15 | #5402 (draft) |

## Phase 2 -- bigger models, more parallelism

### torchtitan (trainer)

Phase-1 default: FSDP2 only, one model copy per trainer node (`dp_replicate` = trainer nodes, `dp_shard` = tiles per
node). Upstream's base recipe: 2 GPUs, TP 2.

| # | Item | XPU now | Status | Upstream |
|---|---|---|---|---|
| 2&#8209;T1 | Tensor parallel | TP 2 on 1 node PASS (upstream Qwen3-0.6B recipe as-is, 10/10); larger models with trainer TP 8 (2-T5a); TP 4 x FSDP multi-node not yet | review 10-26 | -- |
| 2&#8209;T2 | HSDP, 2 copies across nodes | hangs in FSDP2 HSDP on oneCCL (`trn_c2` 2 x 48, both arms, 0 steps, re-confirmed 10-29) | FAILS 10-29 | to file (repro ready; where: user decides) |
| 2&#8209;T3 | Pipeline parallel (#3980); upstream skips per-edge P2P groups on XCCL | not tried | todo | -- |
| 2&#8209;T4 | Context parallel (#5424) | not tried | todo | -- |
| 2&#8209;T6 | Newer architectures: Qwen3.5 / 3.6 (GDN), Kimi K3 (KDA), MXFP8 / NVFP4 | GDN needs NVIDIA-only CuTe kernels; MXFP8 / NVFP4 need SM100 | todo | XPU kernels (torch-xpu-ops / Triton) |
| 2&#8209;T7 | Tasks: DAPO-Math, Search-R1, Verifiers, Terminal-Bench | not tried | todo | -- |
| 2&#8209;T8 | Bitwise parity (batch-invariant kernels) | not tried | todo | -- |
| 2&#8209;T9 | GraphTrainer (`torchtitan/experiments/graph_trainer`): the whole train step as one FX graph, SimpleFSDP, graph passes (comm bucketing, async TP, Inductor regional / full, CUDA graphs, CPU offload); CI is CUDA only; RL does not use it yet | not tried on XPU; needs the latest PyTorch nightly; CUDA graphs and async TP (symmetric memory) need XPU equivalents | todo | torchtitan experiment |

### vLLM (generator)

Phase-1 default: TP 1, one engine per generator tile, data parallel across engines.

| # | Item | XPU now | Status | Upstream |
|---|---|---|---|---|
| 2&#8209;V1 | Generator TP 2 / 4 / 8, also across nodes; weights resharded into TP shards | needs `CCL_SYCL_BROADCAST_SCALEOUT_THRESHOLD=0` (oneCCL #222), set by the launcher | workaround 10-16 | oneCCL #222 |
| 2&#8209;V2 | FSDP generator (#4838) | not on our base | todo | -- (adopt on rebase) |
| 2&#8209;V3 | CUDA / XPU graphs | eager only | todo | -- |
| 2&#8209;V4 | Multi-turn KV hold, sticky routing (#5217-#5220) | not on our base | todo | -- (adopt on rebase) |
| 2&#8209;V5 | Expert / data parallel engines for MoE: TP 4 x DP 2 / EP 8 (Qwen3-30B-A3B), TP 4 / EP 4 (gpt-oss-20b), across nodes | works at 16 nodes with the 0.7 memory limit (2-V6) | workaround 10-29 | -- |
| 2&#8209;V6 | Weight load holds the sharded and the unsharded copy at once; vLLM's memory sizing does not count it | at the default 0.9 of a tile the first forward fails (`UR_RESULT_ERROR_OUT_OF_RESOURCES`); `gpu_memory_limit` 0.7 works | workaround 10-29 | vLLM (profile run under-counts) |

### Larger models (trainer + generator)

Each run is a layout of both sides. Only the runs that pass every check on the current env (2-T5a, 2-T5b) are in
the guide; the rest are planned. Generator side: 2-V1 (TP), 2-V5 (MoE expert / data parallel), 2-V6 (memory).

| # | Item | XPU now | Status | Upstream |
|---|---|---|---|---|
| 2&#8209;T5a | Qwen3-14B, upstream layout: trainer TP 8, one TP-8 engine (2 nodes) | 10/10, every check passes (11-09); **in the guide** | done 11-09 | -- |
| 2&#8209;T5b | Qwen3-1.7B, 16 nodes: 8 x 8 FSDP, 8 TP-8 engines | 10/10, every check passes (11-09); **in the guide** | done 11-09 | -- |
| 2&#8209;T5c | Qwen3-1.7B, upstream layout: trainer TP 2, one TP-4 engine (1 node) | planned: trains 10/10 but fails the logprob-parity check (max 13.3) | todo | -- |
| 2&#8209;T5d | Qwen3-30B-A3B, 16 nodes: 1 x 64 FSDP, 8 TP4 x DP2 / EP8 engines, vLLM at 0.7 | planned: trains 10/10 (~40 min) but fails the logprob-parity check (max 24.4) | todo | -- |
| 2&#8209;T5e | Qwen3-14B, 16 nodes: 8 x 8 FSDP + full remat, 8 TP-8 engines | planned: still training at walltime on the previous stack (~4.5 min/step); re-run on this env 11-09 | todo | -- |
| 2&#8209;T5f | gpt-oss-20b, 16 nodes: 1 x 32 FSDP, 8 TP4 / EP4 engines, vLLM at 0.7 | planned: still training at walltime on the previous stack; re-run on this env 11-09 | todo | -- |
| 2&#8209;T5g | Upstream MoE layouts: 30B-A3B (FSDP 2 x TP 2 x EP 4) and gpt-oss-20b (TP 2 x EP 2 on 2 tiles) | planned: 30B-A3B exits at generator start (oneCCL KVS errors); gpt-oss-20b trainer out of memory | todo | -- |

## Phase 3 -- RDMA on three Intel platforms: CXI (Aurora), BMG and Crescent Island

Three tracks. **3A CXI**: Aurora, PVC (Max 1550, 2 tiles per card, used as 12 `FLAT` tiles per node) + 8 HPE Slingshot
NICs per node, reached through libfabric `cxi`; no ibverbs. **3B BMG**: Intel Arc Battlemage (B-series, one tile per
card, Xe2); network not decided yet, so 3B lists what each NIC type needs. **3C Crescent Island**: Intel Data Center GPU (Xe3P,
inference-focused); specs not known yet, so 3C, like 3B, starts with phases 0-1 on the platform. All use the same plan: weights and
rollouts on PyTorch's `torch.distributed._transport` (11-06: NIXL as the engine, our torch-libfabric as the reference
and fallback). The shared part below has to land once for all; 3A, 3B and 3C add what is platform specific. Upstream order, after
user review: 3A-S4 (Intel NIXL) -> 3-S2 (PyTorch) -> 3-TS1, 3-TS2 (TorchStore) -> 3-T1 (torchtitan); adopt 3-TS3 / 3-S1 when they
land; 3-M1 starts as a Monarch RFC; BMG adds 3B-S4.

### 3.0 Shared by all platforms

| # | Repo | Item | XPU now | Status | Upstream |
|---|---|---|---|---|---|
| 3&#8209;S1 | PyTorch | One `_transport` endpoint per process serving many peers | testing with an overlay of upstream main + #202534; needs a torch nightly with #202535 / #202536 | workaround 11-06 | PyTorch #202534 (open) |
| 3&#8209;S2 | PyTorch | NIXL adapter accepts `xpu` tensors (`nixl/_transport.py:66, 186`; `VRAM` for non-CPU at `:200`), ~3 lines + 1 test; same check in the Mooncake adapter | not written; GRPO weights use the CPU-staged path, so not blocking | todo | PyTorch PR (transport plan T3) |
| 3&#8209;S3 | PyTorch | Stream-ordered `read_stream` / `write_stream` for XPU (today `_cuda_stream.py`, libcuda only): a SYCL queue / event version | not started; optional (callers can sync first) | todo | PyTorch PR |
| 3&#8209;M1 | Monarch | RDMA inside Monarch for XPU: build no longer CUDA / ROCm-gated (`setup.py:218`), Level Zero pointer query + dma-buf export instead of CUDA driver calls (`rdmaxcel-sys`), then a backend per NIC (3A: libfabric `cxi`; 3B: the existing ibverbs backend) | not built on XPU | todo | Monarch RFC / issue first |
| 3&#8209;TS1 | TorchStore | Device-generic stream sync before DMA (`torch.accelerator`; upstream syncs CUDA streams only) | done locally (`torch-transport-device-sync` db4cbc1, on `xpu-all`) | review 11-05 | TorchStore PR (text drafted) |
| 3&#8209;TS2 | TorchStore | Pass backend options (`plugin=...`) to `new_transport` (`torch_distributed.py`) | worked around with a wrapper backend `nixl_lf` | todo | TorchStore PR |
| 3&#8209;TS3 | TorchStore | One NIXL agent per process (was one per transport: cxi abort at 16n x 12) | with #272 + 3-S1 on CXI: GRPO 2n x 4 10/10 pull 0.33 s; 2n x 12 0.86 s; 16n x 12 one copy 10/10 pull 7.68 s (torch-libfabric 3.49 s); 8n x 12 running | workaround 11-06 | TorchStore #272 (open) |
| 3&#8209;TS4 | TorchStore | NIXL and TorchComms transports | TorchComms `RdmaTransport` is CUDA + InfiniBand, uniflow has no XPU adapter; TorchStore's own NIXL transport not tried (NIXL goes through `TorchDistributed`) | todo | -- |
| 3&#8209;TS5 | TorchStore | Direct RDMA: one-hop read of trainer memory (`direct_weight_sync.py`; unused by torchtitan) | needs 3-M1 | todo | -- |
| 3&#8209;TS6 | TorchStore | MonarchRDMA transport | needs 3-M1 | todo | -- |
| 3&#8209;T1 | torchtitan | Rollout microbatches read one-sided (upstream: `MonarchRDMAMicrobatch`, #5285; batches in own process, #5291) | `TransportRDMAMicrobatch` over `_transport`, worktree `~/git/tt_rollout_transport`: on CXI 38 MiB 2.29 ms vs 382 ms (167x); GRPO 2n x 4 with NIXL rollouts + weights trains (E5) | workaround 11-06 | torchtitan PR after 3-S2 |
| 3&#8209;V1 | vLLM | vLLM's own weight-transfer engines (`nccl`, `sparse_nccl`, `nccl_m2n` need PyNccl; `ipc` uses `rebuild_cuda_tensor`) | CUDA only; not needed while torchtitan syncs weights through TorchStore | todo | -- (only if we move to vLLM's engine) |

### 3A. CXI (Aurora: PVC + Slingshot)

Measured: one-sided XPU -> XPU read over cxi 21.2 GB/s, 7 us (XCCL 12 GB/s, 181 us). Required on PVC:
`EnableImplicitScaling=0 NEOReadDebugKeys=1` (with `ZE_FLAT_DEVICE_HIERARCHY=FLAT`), else cxi rejects Level Zero memory.

| # | Repo | Item | XPU now | Status | Upstream |
|---|---|---|---|---|---|
| 3A&#8209;S1 | stack | Feasibility: 2-node `fi_read` XPU -> XPU over cxi; XCCL vs Gloo baseline | XCCL 12 GB/s, Gloo + copies 2.7-4 GB/s; cxi read 21.2 GB/s | done 11-05 | -- |
| 3A&#8209;S2 | stack | torch-libfabric `_transport` backend (reference): C shim + ctypes, one endpoint per process and NIC, native progress thread | tests 9/9 tcp + cxi; 2-node XPU 21.15 GB/s, 12 pairs 77 GB/s | workaround 11-05 | -- (kept local; upstream path is NIXL) |
| 3A&#8209;S3 | NIXL | Level Zero over libfabric `cxi` (no XPU support upstream, no XPU wheel) | Intel's branch `feat/add-intel-xe-*` from source, patched to re-export the bindings like the wheel; stripes every transfer > 128 KB over 8 rails (NUMA-1 rail selection fails) -> tuning | workaround 11-06 | Intel series (#2496 closed by PR-size policy); striping / rail selection to report |
| 3A&#8209;S4 | NIXL | cxi writes without `FI_REMOTE_CQ_DATA` (cxi returns `-FI_EBADFLAGS`; every write failed) | 1-line fix local (`~/git/nixl-intel-cxiwrite` 3b572a10) | review 11-06 | to Intel's NIXL series (note drafted) |
| 3A&#8209;M1 | Monarch | `cxi` backend for `monarch_rdma` (after 3-M1) | not started | todo | Monarch PR, after the RFC |
| 3A&#8209;TS1 | TorchStore | GRPO weights over `TorchDistributed` vs XCCL | torch-libfabric pulls 3.2-3.5x faster (2n x 4 .. 8n x 12), tokens/s 1.3-1.6x, 16n x 12 1 x 96 runs (XCCL cannot), 30B-A3B 4.5x; NIXL 2.8x at 2n x 4 | workaround 11-06 | after 3.0 |

### 3B. BMG (Intel Arc Battlemage)

Nothing run on BMG yet. Before RDMA, BMG needs a pass through phases 0-1 (3B-S1). PVC-only settings do not apply: BMG has
one tile per card, so no `FLAT` / implicit-scaling settings.

| # | Repo | Item | XPU now | Status | Upstream |
|---|---|---|---|---|---|
| 3B&#8209;S1 | all | Phases 0-1 on BMG: torch XPU, `vllm-xpu-kernels` (incl. the SYCL-TLA flash kernel's BMG targets), oneCCL / XCCL, Monarch actors, TorchStore XCCL, GRPO 0.6B smoke | not run; tool chain and regression are Aurora-specific (PBS, PALS, `cxi`) | todo | whatever fails |
| 3B&#8209;S2 | stack | Pick the NIC; then the GPU-direct path. **ibverbs (Mellanox / RoCE)**: Level Zero dma-buf export + `ibv_reg_dmabuf_mr`; engines NIXL UCX or NIXL LIBFABRIC `verbs` (`FI_HMEM_ZE`), torch-libfabric on `verbs`. **One host, several cards**: Level Zero IPC / peer access between cards. **Plain Ethernet**: libfabric `tcp` (same API, TCP speed; no GPU direct) | NIC not decided | todo | -- |
| 3B&#8209;S3 | stack | Feasibility on BMG: 2-node (or 2-card) XPU -> XPU one-sided read with the chosen NIC; XCCL baseline | not run | todo | -- |
| 3B&#8209;S4 | NIXL | Level Zero memory on the chosen plugin (UCX `ze` support or libfabric `verbs` + `FI_HMEM_ZE`); Intel's series covers libfabric only | to check | todo | NIXL / Intel series |
| 3B&#8209;M1 | Monarch | With ibverbs: Monarch's existing ibverbs backend works once 3-M1 adds Level Zero registration; no new backend needed | needs 3-M1 | todo | Monarch PR, after the RFC |
| 3B&#8209;TS1 | TorchStore | Same transports as 3A through 3.0 (3-TS1, 3-TS2, 3-TS3); with ibverbs also MonarchRDMA (3-TS6) after 3B-M1 | -- | todo | -- (covered by 3.0) |
| 3B&#8209;T1 | torchtitan | GRPO on BMG: a launcher for the BMG cluster (not PBS / PALS), rollouts over `_transport` (3-T1) | not started | todo | launcher for that scheduler, if upstream lacks it |


### 3C. Crescent Island (Intel Data Center GPU, Xe3P)

Nothing run on Crescent Island yet. Before RDMA, it needs a pass through phases 0-1 (3C-S1 .. 3C-V1). Specs to
confirm: tiles per card, memory per card, cards per node, node interconnect, scale-out NIC, cluster and
scheduler. Already in the code: vllm-xpu-kernels builds Xe3P / CRI targets when `VLLM_XPU_ENABLE_XE3P` is on (Aurora
builds turn it off).

| # | Repo | Item | XPU now | Status | Upstream |
|---|---|---|---|---|---|
| 3C&#8209;S1 | stack | torch XPU and oneAPI / driver support for Xe3P; `vllm-xpu-kernels` built with its CRI targets; oneCCL / XCCL on CRI | not run | todo | -- |
| 3C&#8209;S2 | stack | Device layout: tiles per card and how they show up (`ZE_FLAT_DEVICE_HIERARCHY`), memory per tile; adjust the launcher's tile counts and thread caps | specs to confirm | todo | -- |
| 3C&#8209;S3 | stack | Network: scale-out NIC and its RDMA path (then 3.0 applies, as for 3B-S2) | specs to confirm | todo | -- |
| 3C&#8209;M1 | Monarch | Actors on CRI devices (the XPU runtime #4307 should apply as is) | not run | todo | -- |
| 3C&#8209;TS1 | TorchStore | XCCL transport on CRI; RDMA transports once 3C-S3 is known | not run | todo | -- |
| 3C&#8209;T1 | torchtitan | GRPO 0.6B smoke and basic on CRI: a launcher for its cluster's scheduler, layouts for its tile count | not started | todo | -- |
| 3C&#8209;V1 | vLLM | Generator on CRI first (an inference part): vLLM XPU worker and kernels on Xe3P, generation correctness | not run | todo | vLLM / vllm-xpu-kernels |

## Platform support

Which Intel GPUs the stack targets and how far each is. Phases 0-2 ran on Aurora only; every other platform starts
with its own pass through phases 0-1.

| Platform | GPU | Where | Network | Status |
|---|---|---|---|---|
| Aurora | Data Center GPU Max 1550 (PVC, Xe-HPC; 2 tiles per card, 12 `FLAT` tiles per node) | ALCF Aurora | HPE Slingshot, libfabric `cxi` | phases 0-2 done; 3A in progress |
| BMG | Arc Battlemage (B-series, Xe2; 1 tile per card) | to confirm | not decided (see 3B) | not started ([3B](#3b-bmg-intel-arc-battlemage)) |
| Crescent&nbsp;Island | Data Center GPU "Crescent Island" (Xe3P; inference-focused, LPDDR5X memory per Intel's announcement) | to confirm | to confirm | not started ([3C](#3c-crescent-island-intel-data-center-gpu-xe3p)) |

## Notes

**Transport choice (phase 0-3).** Upstream TorchStore has 7 transports, tried in this order per (client, volume):
SharedMemory (same host) -> TorchDistributed -> NIXL -> TorchComms -> MonarchRDMA -> Gloo -> MonarchRPC; table and
switches in `~/git/claude-skills/knowledge/torchstore.md` 3.6. Our XCCL (#171) is checked after MonarchRDMA, before Gloo
(`TORCHSTORE_XCCL_ENABLED=1` by default on our branch). On Aurora: SharedMemory for the push, XCCL for every
cross-host pull today (0-TS5), TorchDistributed for phase 3, Gloo and MonarchRPC work but are slow, TorchComms and
MonarchRDMA are unavailable. XCCL keeps tensors on the XPU and sends a batch as one flat buffer; its costs are one
2-rank communicator per pair (1-TS4), a TCPStore handshake per pair (0-TS6) and the oneCCL workarounds (1-T6, 2-V1).

**Rollout transport (1-M2, 3-T1).** Every rollout hop is a Monarch endpoint call or reply; only batcher -> trainer is
bulky. Upstream: the default Monarch message, or opt-in `MonarchRDMAMicrobatch` (`~/git/claude-skills/knowledge/torchtitan.md` 3.5). On Aurora the
first runs on the management network until 1-M2, the second is unavailable; ours (3-T1) reads over `_transport`.

**TorchStore on Monarch.** Monarch is TorchStore's control path (actors, endpoint calls, handshakes) and its data
path only for MonarchRPC and MonarchRDMA: `~/git/claude-skills/knowledge/torchstore.md` section 2.

**Why no Monarch RDMA (3-M1, 3-TS5, 3-TS6).** Monarch builds RDMA only with a CUDA / ROCm toolchain
(`build_rdma = has_cuda or has_rocm`), registers GPU memory with CUDA driver calls only, and its NIC backends are
ibverbs (mlx5, EFA) plus a TCP fallback. Aurora compute nodes have 8 Slingshot Cassini NICs (`cxi0-7`) and no
ibverbs; libfabric 2.3.1 `cxi` offers `FI_RMA`, `FI_READ`, `FI_HMEM` (Level Zero), 24.2 GB/s vs tcp 3.0 GB/s on one
NIC. oneCCL already uses cxi (two-sided collectives); Gloo and Monarch messages use TCP. Details:
`~/git/claude-skills/xpu_rl/transport_experiments_results.md`.
