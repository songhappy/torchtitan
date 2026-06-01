# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified approach for running TorchTitan models with vLLM inference.

To register TorchTitan models with vLLM:
    from torchtitan.components.checkpoint import CheckpointManager
    from torchtitan.experiments.rl.models.vllm_registry import registry_to_vllm

    # Standalone inference (loads HF weights):
    registry_to_vllm(
        model_spec,
        parallelism=parallelism_config,
        compile_config=compile_config,
        checkpoint_config=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            initial_load_path="/path/to/hf/checkpoint",
        ),
    )

    # RL loop (skip HF loading, weights from TorchStore):
    registry_to_vllm(
        model_spec,
        parallelism=parallelism_config,
        compile_config=compile_config,
        checkpoint_config=CheckpointManager.Config(enable=False),
    )
"""

"""Lazy-import vLLM-related symbols.

Importing :mod:`torchtitan.experiments.rl.models.vllm_wrapper` requires
the ``vllm`` package, which is not installed on every device backend
(notably Intel XPU has no upstream vLLM wheel as of writing). Loading
the wrapper eagerly therefore prevents anything in this experiments
package from importing on XPU. The ``__getattr__`` shim keeps the
public API (``VLLMModelWrapper``, ``registry_to_vllm``) stable while
deferring the import to first use.
"""


__all__ = [
    "VLLMModelWrapper",
    "registry_to_vllm",  # Export register function for manual use
]


def __getattr__(name):
    if name == "VLLMModelWrapper":
        from torchtitan.experiments.rl.models.vllm_wrapper import VLLMModelWrapper
        return VLLMModelWrapper
    if name == "registry_to_vllm":
        from torchtitan.experiments.rl.models.vllm_registry import registry_to_vllm
        return registry_to_vllm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
