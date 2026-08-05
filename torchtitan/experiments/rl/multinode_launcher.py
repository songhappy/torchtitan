# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Multi-node RL training launcher for Monarch on XPU (PBS/mpiexec).

mpiexec launches one rank per node. Each rank starts a Monarch worker.
Rank 0 additionally runs the RL controller that attaches to all workers
and orchestrates GRPO training.

Configurable: number of nodes (from PBS) and GPUs per node (--gpus_per_node).
Parallelism auto-scales: nodes split evenly between trainer and generator,
dp_shard and dp fill available GPUs.

Usage:
    mpiexec -n $NUM_NODES -ppn 1 --hosts $ALL_NODES --envall \
        python3 -m torchtitan.experiments.rl.multinode_launcher \
        --num_nodes=2 --gpus_per_node=4 \
        --module alphabet_sort --config rl_grpo_lora_qwen3_0_6b \
        --hf_assets_path=/path/to/model
"""

import argparse
import asyncio
import logging
import os
import socket
import sys
import threading
import time

# expandable_segments corrupts XPU's oneCCL USM pointers.
if "ZE_AFFINITY_MASK" not in os.environ:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger = logging.getLogger(__name__)

MONARCH_PORT = 26600


def get_mpi_rank() -> int:
    """Get MPI rank from environment (Cray PALS, PMI, or MPICH)."""
    for var in ("PALS_RANKID", "PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK"):
        if var in os.environ:
            return int(os.environ[var])
    return 0


def _configure_timeouts() -> None:
    """Increase Monarch spawn timeouts for multi-node XPU (device init is slow)."""
    from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
    from monarch._rust_bindings.monarch_hyperactor.config import configure

    configure(
        default_transport=ChannelTransport.TcpWithHostname,
        host_spawn_ready_timeout="300s",
        mesh_proc_spawn_max_idle="300s",
    )


def run_worker(address: str) -> None:
    """Start Monarch worker loop (blocks forever)."""
    from monarch.actor import run_worker_loop_forever

    _configure_timeouts()
    logger.info(f"Monarch worker starting on {address}")
    run_worker_loop_forever(address=address, ca="trust_all_connections")


async def run_controller(
    worker_addrs: list[str],
    num_nodes: int,
    gpus_per_node: int,
    train_argv: list[str],
) -> None:
    """Attach to all workers and run the RL training loop.

    Auto-scales parallelism to fill allocated resources:
    - Nodes split evenly: first half trainer, second half generator
    - Trainer dp_shard = trainer_gpus / (TP * PP * CP)
    - Generator dp = generator_gpus / gen_TP
    """
    from dataclasses import replace

    from monarch._src.actor.bootstrap import attach_to_workers

    from torchtitan.config import ConfigManager
    from torchtitan.experiments.rl.controller import Controller
    from torchtitan.experiments.rl.train import (
        HostMeshes,
        _compute_generator_world_size,
        _compute_trainer_world_size,
        breakable_cudagraph_env,
        spawn_proc_mesh,
    )
    from torchtitan.observability import structured_logger as sl

    os.environ["MONARCH_ACTOR_QUEUE_DISPATCH"] = "0"

    # Parse the train config using the same CLI as train.py
    sys.argv = ["train"] + train_argv
    config = ConfigManager().parse_args()
    assert isinstance(config, Controller.Config)

    sl.init_structured_logger(
        source="rl_controller",
        output_dir=config.dump_folder,
        rank=0,
        enable=config.trainer.debug.enable_structured_logging,
    )

    logger.info(f"Attaching to {num_nodes} workers: {worker_addrs}")
    host_mesh = attach_to_workers(
        name="grpo_xpu",
        ca="trust_all_connections",
        workers=worker_addrs,
    )

    # Split nodes: half trainer, half generator. For odd counts, give
    # the extra node to generator (more generation throughput is better).
    num_trainer_nodes = num_nodes // 2 or 1
    num_generator_nodes = num_nodes - num_trainer_nodes

    trainer_host_mesh = host_mesh.slice(hosts=slice(0, num_trainer_nodes))
    generator_host_mesh = host_mesh.slice(
        hosts=slice(num_trainer_nodes, num_nodes)
    )
    host_meshes = HostMeshes(
        trainer=trainer_host_mesh,
        generators=[generator_host_mesh],
        gpus_per_node=gpus_per_node,
    )

    # Scale parallelism to fill allocated GPUs.
    # Cap dp_shard at the LoRA rank (if LoRA is used) to avoid zero-sized FSDP
    # shards that TorchStore cannot handle. Excess GPUs go to dp_replicate.
    trainer_total_gpus = num_trainer_nodes * gpus_per_node
    tp = config.trainer.parallelism.tensor_parallel_degree
    pp = config.trainer.parallelism.pipeline_parallel_degree
    cp = config.trainer.parallelism.context_parallel_degree
    dp_replicate = config.trainer.parallelism.data_parallel_replicate_degree
    trainer_dp_shard = trainer_total_gpus // (tp * pp * cp * dp_replicate)

    # Detect LoRA rank from the model config tree. After LoRA conversion,
    # target Linear configs become LoRALinear.Config with `rank` + `alpha`.
    def _find_lora_rank(obj, depth=0):
        if depth > 5:
            return None
        if hasattr(obj, "rank") and hasattr(obj, "alpha"):
            return obj.rank
        if isinstance(obj, (list, tuple)):
            for item in obj:
                r = _find_lora_rank(item, depth + 1)
                if r is not None:
                    return r
        elif hasattr(obj, "__dataclass_fields__"):
            for field_name in obj.__dataclass_fields__:
                r = _find_lora_rank(getattr(obj, field_name), depth + 1)
                if r is not None:
                    return r
        return None

    # FSDP shards each param on dim 0, and the smallest LoRA tensor dim is the
    # LoRA rank, so dp_shard can grow up to the rank without producing zero-sized
    # shards (which TorchStore cannot handle). Cap at the rank -- NOT a hardcoded
    # 4 -- so all trainer GPUs go to dp_shard (dp_replicate stays 1) whenever the
    # rank allows; only spill to dp_replicate past that.
    lora_rank = _find_lora_rank(config.model_spec.model)
    max_dp_shard = lora_rank or 4
    if trainer_dp_shard > max_dp_shard:
        dp_replicate = trainer_dp_shard // max_dp_shard
        trainer_dp_shard = max_dp_shard
        logger.info(
            f"Capping dp_shard at {max_dp_shard}, "
            f"dp_replicate={dp_replicate}"
        )

    # The requested dp_replicate x dp_shard split is used as-is, including when a
    # replicate group spans physical nodes.
    #
    # There used to be an auto-fold here that rewrote a cross-node replicate group
    # into dp_shard (dp_shard *= dp_replicate, dp_replicate = 1). It was removed on
    # 2026-07-31 after both of its premises were measured false; do not re-add it:
    #   1. CORRECTNESS. Cross-node dp_replicate looked broken on this Monarch/XPU
    #      (xccl/CXI) stack -- dp_shard4/rep2 exploded in bit_wise/logprob_diff and
    #      dp_shard1/rep2 x tp4 showed grad_norm 141-1352. Both were the CXI/oneCCL
    #      scale-out env, not the mesh: with that env and NO mesh change the same
    #      shapes train clean at grad_norm 0.058-0.17.
    #   2. PERF. The fold was then kept as the faster layout, but the unfolded runs
    #      are faster on identical allocations: dp_shard4 x rep2 7163 vs 6529 tok/s,
    #      rep2 x tp4 957 vs 900. dp_replicate all-reduces gradients once per step
    #      while dp_shard all-gathers parameters every layer, so folding cross-node
    #      DP onto dp_shard only adds fabric traffic.
    config.trainer.parallelism = replace(
        config.trainer.parallelism,
        data_parallel_shard_degree=trainer_dp_shard,
        data_parallel_replicate_degree=dp_replicate,
    )

    generator_total_gpus = num_generator_nodes * gpus_per_node
    gen_tp = config.generator.parallelism.tensor_parallel_degree
    generator_dp = generator_total_gpus // gen_tp
    config.generator.parallelism = replace(
        config.generator.parallelism,
        data_parallel_degree=generator_dp,
    )

    logger.info(
        f"Mesh split: {num_trainer_nodes} trainer node(s) "
        f"({trainer_total_gpus} GPUs, dp_shard={trainer_dp_shard}, tp={tp}), "
        f"{num_generator_nodes} generator node(s) "
        f"({generator_total_gpus} GPUs, dp={generator_dp}, tp={gen_tp})"
    )

    rl_trainer: Controller = config.build()
    try:
        trainer_world_size = _compute_trainer_world_size(config.trainer.parallelism)
        per_generator_world_size = _compute_generator_world_size(
            config.generator.parallelism
        )
        trainer_mesh, generator_meshes = spawn_proc_mesh(
            trainer_world_size,
            per_generator_world_size,
            host_meshes=host_meshes,
            num_generators=config.num_generators,
            generator_env=breakable_cudagraph_env(config.generator),
        )
        await rl_trainer.setup_async(
            trainer_mesh=trainer_mesh,
            generator_meshes=generator_meshes,
        )
        await rl_trainer.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted; attempting graceful shutdown...")
    finally:
        await rl_trainer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-node GRPO launcher",
        allow_abbrev=False,
    )
    parser.add_argument("--num_nodes", type=int, required=True)
    parser.add_argument("--gpus_per_node", type=int, default=4)
    parser.add_argument(
        "--all_nodes",
        type=str,
        required=True,
        help="Comma-separated list of all hostnames (from PBS_NODEFILE)",
    )
    args, train_argv = parser.parse_known_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    rank = get_mpi_rank()
    hostname = socket.gethostname()

    # Strip PMI/PALS env vars so spawned child procs don't try to use
    # mpiexec's PMI wire protocol for torch.distributed rendezvous.
    for var in list(os.environ):
        if var.startswith(("PMI_", "PALS_")):
            del os.environ[var]

    # Force oneCCL to use OFI (libfabric) transport instead of MPI.
    # Without PMI vars, all spawned procs appear as MPI rank 0, causing
    # MPI_Group_incl to abort on duplicate ranks.
    os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")

    # Use Cray's libfabric (has CXI provider for native Slingshot RDMA).
    # Conda's libfabric only has tcp/psm2/psm3 -- no CXI support.
    cray_libfabric = "/opt/cray/libfabric/1.22.0/lib64/libfabric.so.1"
    if os.path.exists(cray_libfabric):
        os.environ.setdefault("CCL_OFI_LIBRARY_PATH", cray_libfabric)
        os.environ.setdefault("FI_PROVIDER", "cxi")
        os.environ.setdefault("CCL_ATL_OFI_PROVIDER", "cxi")

    all_nodes = list(dict.fromkeys(args.all_nodes.split(",")))

    # Worker address must use the same name form as all_nodes so that the
    # worker identity matches what attach_to_workers uses. Bind on 0.0.0.0
    # so spawned child procs can connect back via any interface.
    my_addr_name = hostname
    for node in all_nodes:
        if node.startswith(hostname):
            my_addr_name = node
            break

    logger.info(f"Rank {rank} on {hostname}, nodes={all_nodes}")

    worker_addr = (
        f"tcp://{my_addr_name}:{MONARCH_PORT}@tcp://0.0.0.0:{MONARCH_PORT}"
    )

    _configure_timeouts()

    if rank == 0:
        worker_thread = threading.Thread(
            target=run_worker, args=(worker_addr,), daemon=True
        )
        worker_thread.start()
        time.sleep(2)

        worker_addrs = [f"tcp://{node}:{MONARCH_PORT}" for node in all_nodes]
        try:
            asyncio.run(
                run_controller(
                    worker_addrs, args.num_nodes, args.gpus_per_node, train_argv
                )
            )
        except Exception:
            logger.exception("Controller crashed")
            sys.exit(1)
    else:
        run_worker(worker_addr)


if __name__ == "__main__":
    main()
