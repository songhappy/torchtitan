# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest

from torchtitan.rl.multinode_launcher import _derive_dp_degrees


def derive(
    trainer_total_gpus: int,
    *,
    gpus_per_node: int,
    tp: int = 1,
    pp: int = 1,
    cp: int = 1,
    dp_replicate: int = 1,
    lora_rank: int | None = None,
) -> tuple[int, int]:
    """_derive_dp_degrees with the launcher's defaults filled in."""
    return _derive_dp_degrees(
        trainer_total_gpus,
        gpus_per_node=gpus_per_node,
        tp=tp,
        pp=pp,
        cp=cp,
        dp_replicate=dp_replicate,
        lora_rank=lora_rank,
    )


def test_shard_group_stays_within_one_node() -> None:
    # 8 nodes split half trainer / half generator, so 4 trainer nodes.
    assert derive(4 * 4, gpus_per_node=4) == (4, 4)
    assert derive(4 * 8, gpus_per_node=8) == (4, 8)
    # 2 nodes: one trainer node, so a single replica already fits.
    assert derive(8, gpus_per_node=8) == (1, 8)


@pytest.mark.parametrize("gpus_per_node", range(1, 13))
@pytest.mark.parametrize("num_trainer_nodes", (1, 2, 3, 4, 8))
@pytest.mark.parametrize("lora_rank", (None, 32))
def test_every_supported_tile_count_yields_a_usable_mesh(
    gpus_per_node: int, num_trainer_nodes: int, lora_rank: int | None
) -> None:
    """1 to 12 tiles per node is the whole supported range (an Aurora node has 12).

    Every combination has to cover the allocation exactly and keep the shard
    group inside a node; a gap either trains on part of the allocation or pays
    cross-node all-gathers every layer.
    """
    total = num_trainer_nodes * gpus_per_node
    dp_replicate, dp_shard = derive(
        total, gpus_per_node=gpus_per_node, lora_rank=lora_rank
    )
    assert dp_replicate * dp_shard == total
    assert dp_shard <= gpus_per_node


def test_lora_rank_lowers_the_cap_but_never_raises_it() -> None:
    # Rank above one node's tiles: locality binds, same as full parameter.
    assert derive(4 * 8, gpus_per_node=8, lora_rank=32) == (4, 8)
    # Rank below it: the zero-sized-shard bound binds instead.
    assert derive(4 * 8, gpus_per_node=8, lora_rank=2) == (16, 2)


def test_explicit_dp_replicate_composes_with_the_cap() -> None:
    # Asking for exactly the locality-capped layout leaves it untouched.
    assert derive(4 * 8, gpus_per_node=8, dp_replicate=4) == (4, 8)
    # Asking for less replication than the cap implies still fills the mesh:
    # the spill multiplies into the requested degree rather than replacing it.
    assert derive(4 * 8, gpus_per_node=8, dp_replicate=2) == (4, 8)


def test_tensor_parallel_shrinks_the_shard_group() -> None:
    # tp rides inside the shard group's node, so dp_shard * tp <= gpus_per_node.
    dp_replicate, dp_shard = derive(4 * 8, gpus_per_node=8, tp=2)
    assert (dp_replicate, dp_shard) == (4, 4)
    assert dp_shard * 2 == 8


def test_degrees_that_cannot_cover_the_allocation_raise() -> None:
    with pytest.raises(ValueError, match="trainer GPUs"):
        derive(12, gpus_per_node=4, tp=8)
