# XPU RL roadmap: GRPO on Intel XPU vs upstream CUDA

The gaps between upstream torchtitan RL (async GRPO: Monarch, TorchStore, vLLM, FSDP2) on CUDA and on Intel XPU, and
their status. Build, run and validate: [xpu_torch215.md](xpu_torch215.md). Updated 2026-11-10.

| Phase | Goal |
|---|---|
| **0** | each component works on its own on XPU |
| **1** | Qwen3-0.6B GRPO end to end, 1-16 nodes |
| **2** | enable bigger models, more parallelism and advanced trainer/vllm features (trainer and generator) |
| **3** | RDMA weight and rollout transfer: 3A CXI (Aurora), 3B BMG, 3C Crescent Island |
| **4** | Demonstrate numerical parity and competitive performance for RL using bigger models |


IDs are `<phase>-<repo><n>` (S stack, M Monarch, TS TorchStore, T torchtitan, V vLLM). Status: `done`,
`workaround`, `review` (done, PR open or local), `FAILS`, `todo`; dates 2026.

**State:** Qwen3-0.6B GRPO runs on Aurora up to 16 nodes x 12 tiles; Monarch's XPU support is upstream; TorchStore and
torchtitan PRs are in review; larger models run with workarounds; one-sided transfer works on Aurora (3A).

## Phase 0 -- each component on XPU

| # | Item | XPU now | Status | Upstream |
|---|---|---|---|---|
| 0&#8209;S1 | PyTorch with XPU | torch 2.15.0, triton-xpu 3.9.0 | done 11-09 | -- |
| 0&#8209;S2 | oneAPI / oneCCL | oneAPI 2026.1, oneCCL 2022.1.2 | done 11-09 | -- |
| 0&#8209;S3 | RMSNorm backward, large batches | fixed | done 09-14 | torch-xpu-ops #4790 (merged) |
| 0&#8209;S4 | oneCCL bugs: broadcast scale-out, worker affinity, concurrent communicator creation | workarounds (1-T27, 1-T6, 2-V1) | workaround | oneCCL #222-#225 (open) |
| 0&#8209;S5 | Segfault at exit after an XCCL group | harmless for GRPO | FAILS 11-04 | to file |
| 0&#8209;S6 | Same torch / vLLM as upstream main | vLLM main drops config torchtitan passes; we stay on vLLM `083060d04` | todo | torchtitan generator |
| 0&#8209;M1 | Monarch build without the tensor engine | builds | done 10-05 | -- |
| 0&#8209;M2 | Monarch actors on XPU devices | works | done 11-03 | #4307 (merged) |
| 0&#8209;M3 | Monarch on PBS clusters (`PBSJob`) | works | done 10-29 | #5067 (merged) |
| 0&#8209;TS1 | TorchStore on XPU, on upstream main | works | done 11-03 | -- |
| 0&#8209;TS3 | Host name from `TORCHSTORE_HOSTNAME` (#226) | not adopted | todo | -- (upstream has it) |
| 0&#8209;TS4 | XCCL transport (CUDA uses RDMA), SharedMemory XPU-safe, handshake port race | works | review 11-03 | #171 (open), #221 (draft) |
| 0&#8209;TS5 | Gloo, MonarchRPC transports | work, slower than XCCL | done | -- |
| 0&#8209;V1 | vLLM + `vllm-xpu-kernels` on XPU | works | done 10-05 | -- |
| 0&#8209;V2 | Generation correctness, TP 1 / 2 / 4 | correct; TP > 1 dumps core at exit | review 10-26 | -- |

## Phase 1 -- Qwen3-0.6B GRPO end to end

| # | Item | XPU now | Status | Upstream |
|---|---|---|---|---|
| 1&#8209;M1 | Rollouts through controller, router, actors (96 generators) | works | done | -- |
| 1&#8209;M2 | Monarch messages on the fast network (`HYPERACTOR_TCP_INTERFACE`; upstream binds the hostname, ~1 Gb) | 6.6x on 38 MiB | review 11-09 | Monarch PR (local) |
| 1&#8209;TS1 | Default client (storage volumes) | works | done | -- |
| 1&#8209;TS2 | Routing client; each rank reads one replica on XCCL | works | review | PR draft |
| 1&#8209;TS3 | Fetch each replicated region once | works | review 11-03 | #201 (open) |
| 1&#8209;TS4 | oneCCL caps a process at 128 communicators (one per pair) | larger one-copy layouts fail; phase 3 removes it | FAILS | -- |
| 1&#8209;T24 | RFC: torchtitan RL on XPU | posted | review 10-15 | #5401 |
| 1&#8209;T1 | GRPO, full + LoRA, flex attention, HF weights, FSDP2 (one node, one copy per node, one copy across nodes) | works, 200 steps | done 10-16 | -- |
| 1&#8209;T25 | Aurora launcher: PALS, CXI / oneCCL env, thread caps, node-local compile cache | works | done 10-14 | -- (site script) |
| 1&#8209;T27 | oneCCL 2022 settings (`CCL_OP_SYNC`, `CCL_WORKER_AFFINITY` unset) | works | workaround 10-06 | oneCCL #223 |
| 1&#8209;T28 | Device visibility of spawned processes (`ZE_AFFINITY_MASK`) | works | review 10-15 | #5402 (draft, needs rebase) |
| 1&#8209;T4 | Multi-host driver on a Monarch job (Slurm or PBS) | works | review 10-15 | #5403 (draft; rework on upstream launchers #5572) |
| 1&#8209;T6 | oneCCL #222 / #225 workarounds, on for XPU | works | workaround 10-15 | #5404 (draft) |
| 1&#8209;T7 | Allocator safe for transfers (no `expandable_segments` on XPU) | works | review 10-15 | #5402 (draft) |
| 1&#8209;T11 | FSDP2, one copy over 96 tiles | works on torch 2.15 | workaround 10-29 | -- |
| 1&#8209;T14 | Scale: 1-16 nodes, 1-12 tiles per node | works, except 1-TS4 and 2-T2 | done 10-16 | -- |
| 1&#8209;T15 | Save checkpoint | off (load only) | todo | -- |
| 1&#8209;T16 | Prompt data state in checkpoints (#5114) | not on our base | todo | adopt on rebase |
| 1&#8209;T17 | Keep up with upstream main | rebased 10-28; main moved since | workaround | before #5402-#5404 land |
| 1&#8209;T18 | `torch.accelerator` for device / stream / memory (#5450) | not on our base | todo | adopt on rebase |
| 1&#8209;T19 | Weights staged in pinned CPU memory (#4945, #5107) | not on our base | todo | adopt on rebase |
| 1&#8209;V1 | vLLM in the RL loop: eager, XPU attention, `XPUWorker` (upstream: CUDA graphs, CUDA-only worker) | works, 96 generators | review 10-15 | #5402 (draft) |

## Phase 2 -- bigger models, more parallelism, advanced trainer and vLLM features

Phase-1 defaults: trainer FSDP2, one copy per node; generators one TP-1 engine per tile.

| # | Item | XPU now | Status | Upstream |
|---|---|---|---|---|
| 2&#8209;T1 | Trainer tensor parallel | TP 2 and TP 8 work; TP x FSDP across nodes not yet | review 10-26 | -- |
| 2&#8209;T2 | HSDP, 2 copies across nodes | hangs (FSDP2 on oneCCL) | FAILS 10-29 | to file |
| 2&#8209;T3 | Pipeline parallel (#3980) | not tried | todo | -- |
| 2&#8209;T4 | Context parallel (#5424) | not tried | todo | -- |
| 2&#8209;T6 | Newer architectures: GDN, KDA, MXFP8 / NVFP4 | need NVIDIA-only kernels | todo | XPU kernels |
| 2&#8209;T7 | Tasks: DAPO-Math, Search-R1, Verifiers, Terminal-Bench | not tried | todo | -- |
| 2&#8209;T9 | GraphTrainer (whole step as one FX graph; CUDA-only CI) | not tried | todo | torchtitan experiment |
| 2&#8209;V1 | Generator TP 2 / 4 / 8, also across nodes | works with a oneCCL setting | workaround 10-16 | oneCCL #222 |
| 2&#8209;V2 | FSDP generator (#4838) | not on our base | todo | adopt on rebase |
| 2&#8209;V3 | CUDA / XPU graphs in vLLM | eager only | todo | -- |
| 2&#8209;V4 | Multi-turn KV hold, sticky routing (#5217-#5220) | not on our base | todo | adopt on rebase |
| 2&#8209;V5 | Expert / data parallel engines for MoE, across nodes | works with 2-V6 | workaround 10-29 | -- |
| 2&#8209;V6 | Weight load holds two copies; vLLM's memory sizing misses it | works at 0.7 of a tile | workaround 10-29 | vLLM |
| 2&#8209;T5a | Qwen3-14B, upstream layout (2 nodes) | works; in the guide | done 11-09 | -- |
| 2&#8209;T5b | Qwen3-1.7B, 16 nodes | works; in the guide | done 11-09 | -- |
| 2&#8209;T5e | Qwen3-14B, 16 nodes | works | done 11-09 | -- |
| 2&#8209;T5f | gpt-oss-20b, 16 nodes | tokenizer vocab not cached on compute nodes | todo | -- |
| 2&#8209;T5g | Upstream MoE layouts (30B-A3B, gpt-oss-20b) | oneCCL errors; out of memory | todo | -- |

## Phase 3 -- RDMA on three Intel platforms: CXI (Aurora), BMG and Crescent Island

One plan for all platforms (hardware: [Platform support](#platform-support)): weights and rollouts over PyTorch's
`torch.distributed._transport`, NIXL as the engine, our torch-libfabric as the reference. Upstream order: 3A-S4 ->
3-S2 -> 3-TS1, 3-TS2 -> 3-T1; 3-M1 starts as a Monarch RFC.

### 3.0 Shared by all platforms

| # | Repo | Item | XPU now | Status | Upstream |
|---|---|---|---|---|---|
| 3&#8209;S1 | PyTorch | One `_transport` endpoint per process | works with #202534 | workaround 11-06 | #202534 (open) |
| 3&#8209;S2 | PyTorch | NIXL adapter accepts XPU tensors | done locally | review 11-09 | PyTorch PR (local) |
| 3&#8209;S3 | PyTorch | Stream-ordered transport ops for XPU | optional | todo | PyTorch PR |
| 3&#8209;M1 | Monarch | RDMA in Monarch for XPU (build without CUDA, Level Zero memory, a backend per NIC) | not built on XPU | todo | Monarch RFC |
| 3&#8209;TS1 | TorchStore | Stream sync before DMA on any accelerator | done locally | review 11-05 | TorchStore PR (local) |
| 3&#8209;TS2 | TorchStore | Backend options for `torch.distributed._transport` | done locally | review 11-09 | TorchStore PR (local) |
| 3&#8209;TS3 | TorchStore | One NIXL agent per process | works with #272 | workaround 11-06 | #272 (open) |
| 3&#8209;TS4 | TorchStore | Its own NIXL and TorchComms transports | TorchComms is CUDA + InfiniBand | todo | -- |
| 3&#8209;TS5 | TorchStore | Direct RDMA, MonarchRDMA transport | need 3-M1 | todo | -- |
| 3&#8209;T1 | torchtitan | Rollout microbatches read one-sided (upstream: Monarch RDMA) | works, 167x on 38 MiB | workaround 11-06 | torchtitan PR after 3-S2 |
| 3&#8209;V1 | vLLM | vLLM's own weight-transfer engines | CUDA only; not needed | todo | -- |

### 3A. CXI (Aurora)

Requires `EnableImplicitScaling=0 NEOReadDebugKeys=1`, else cxi rejects Level Zero memory.

| # | Repo | Item | XPU now | Status | Upstream |
|---|---|---|---|---|---|
| 3A&#8209;S1 | stack | One-sided XPU -> XPU read over cxi | 21 GB/s (XCCL 12) | done 11-05 | -- |
| 3A&#8209;S2 | stack | torch-libfabric `_transport` backend (reference) | works, 21 GB/s per NIC | workaround 11-05 | -- (local) |
| 3A&#8209;S3 | NIXL | Level Zero over libfabric `cxi` | Intel's series, from source; known issue: writes ~1.5x slower than torch-libfabric at 552 KB (GRPO's size), faster above 1 MiB | workaround 11-06 | Intel series (unmerged) |
| 3A&#8209;S4 | NIXL | cxi writes fail (`FI_REMOTE_CQ_DATA`) | 1-line fix, local | review 11-06 | Intel series |
| 3A&#8209;M1 | Monarch | `cxi` backend for Monarch RDMA | after 3-M1 | todo | Monarch PR |
| 3A&#8209;TS1 | TorchStore | GRPO weights over `_transport` vs XCCL | 2n x 4 pull: torch-libfabric 0.29 s, NIXL 0.32 s, XCCL 0.95 s; 16n x 12 one copy runs (XCCL cannot) | workaround 11-09 | after 3.0 |

### 3B. BMG (Intel Arc Battlemage)

| # | Repo | Item | XPU now | Status | Upstream |
|---|---|---|---|---|---|
| 3B&#8209;S1 | all | Phases 0-1 on BMG | not run | todo | -- |
| 3B&#8209;S2 | stack | Pick the NIC (ibverbs, Level Zero IPC, or Ethernet), then its GPU-direct path | not decided | todo | -- |
| 3B&#8209;S3 | stack | One-sided XPU -> XPU read with that NIC | not run | todo | -- |
| 3B&#8209;S4 | NIXL | Level Zero memory on that NIC's plugin | to check | todo | NIXL |
| 3B&#8209;M1 | Monarch | Monarch's ibverbs backend with Level Zero memory | after 3-M1 | todo | Monarch PR |
| 3B&#8209;T1 | torchtitan | GRPO on BMG: launcher for its scheduler | not started | todo | -- |

### 3C. Crescent Island (Intel Data Center GPU, Xe3P)

| # | Repo | Item | XPU now | Status | Upstream |
|---|---|---|---|---|---|
| 3C&#8209;S1 | stack | torch, oneAPI, `vllm-xpu-kernels` (CRI targets), oneCCL on Xe3P | not run | todo | -- |
| 3C&#8209;S2 | stack | Device layout and launcher tile counts | specs to confirm | todo | -- |
| 3C&#8209;S3 | stack | Scale-out NIC and its RDMA path | specs to confirm | todo | -- |
| 3C&#8209;M1 | Monarch | Actors on Crescent Island | not run | todo | -- |
| 3C&#8209;TS1 | TorchStore | XCCL, then RDMA transports | not run | todo | -- |
| 3C&#8209;T1 | torchtitan | GRPO 0.6B smoke and basic regression | not started | todo | -- |
| 3C&#8209;V1 | vLLM | Generator on Xe3P (an inference part) | not run | todo | vLLM / vllm-xpu-kernels |

## Phase 4 -- numerical parity and competitive performance (bigger models)

| # | Item | XPU now | Status | Upstream |
|---|---|---|---|---|
| 4&#8209;T1 | Logprob parity, trainer vs generator, on the larger models | mean small; max > 10 (4-T1a, 4-T1b); cause unknown | todo | -- |
| 4&#8209;T1a | Qwen3-1.7B, upstream layout (1 node) | trains; logprob-parity max 13.3 | todo | -- |
| 4&#8209;T1b | Qwen3-30B-A3B, 16 nodes | trains; logprob-parity max 24.4 | todo | -- |
| 4&#8209;T2 | Bitwise parity (batch-invariant kernels) | not tried | todo | -- |
| 4&#8209;T3 | Training throughput: 8-node full ~30% slower than on torch 2.12 | not investigated | todo | -- |
| 4&#8209;T4 | Training and generation throughput vs CUDA, same models and layouts | not measured | todo | -- |

## Platform support

| Platform | GPU | Network | Status |
|---|---|---|---|
| Aurora | Max 1550 (PVC, Xe-HPC), 12 tiles per node | 8 Slingshot NICs, libfabric `cxi`; no ibverbs | phases 0-2 done; 3A in progress |
| BMG | Arc Battlemage (Xe2), 1 tile per card | not decided | not started |
| Crescent&nbsp;Island | Xe3P, inference-focused, LPDDR5X | to confirm | not started |
