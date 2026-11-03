# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.attention import (
    activate_flash_attention_impl,
    current_flash_attention_impl,
)
from torch.nn.attention.varlen import AuxRequest
from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import warn_once
from torchtitan.tools.utils import has_cuda_capability
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

logger = logging.getLogger(__name__)


@register_backend(AttentionBackendEnum.CUSTOM)
class PyTorchVarlenAttentionBackend(FlashAttentionBackend):
    """Custom vLLM attention backend using PyTorch's native FlashAttention kernel.

    This class is not directly referenced in user code. It is registered into
    vLLM's attention backend registry via the ``@register_backend`` decorator
    and selected at runtime when the vLLM engine is configured to use a CUSTOM
    attention backend.

    Inheriting from ``FlashAttentionBackend`` is not strictly required for all
    backends, but it is convenient here to reuse metadata construction logic.
    """

    @staticmethod
    def get_name():
        # vLLM requires any custom attention backend to return "CUSTOM" as its
        # name so the backend registry can look it up correctly.
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return PyTorchVarlenAttentionImpl

    @staticmethod
    def get_builder_cls():
        # Report UNIFORM_SINGLE_TOKEN_DECODE cudagraph support instead of the FA3
        # builder's ALWAYS. Our varlen forward bakes per-step cu_seqlens /
        # max_query_len into the captured graph, so a FULL graph over a mixed
        # prefill+decode batch replays stale offsets -> NaN (#3709); only
        # query_len==1 decode is safe to capture. This keeps FULL_DECODE_ONLY
        # valid, auto-downgrades FULL to FULL_DECODE_ONLY (instead of capturing the
        # broken mixed graph), and allows PIECEWISE (attention runs eager).
        class PyTorchVarlenAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
            _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

        return PyTorchVarlenAttentionMetadataBuilder


class PyTorchVarlenAttentionImpl(FlashAttentionImpl):
    """
    Custom vLLM attention backend impl using PyTorch's native FlashAttention varlen API.
    Instead of using vLLM's FlashAttention kernel, this implementation takes the kernel
    dependency from torch directly while supporting the same interface.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # optional post-attention epilogue transform (out, lse) -> out
        # set by VLLMModelWrapper in vllm_wrapper.py
        self.out_transform = None

        self.enable_gqa = self.num_heads > self.num_kv_heads

        # Hopper (SM 9.0) uses FA3
        if has_cuda_capability(9, 0):
            # activate_flash_attention_impl() will restore internal global state
            # and re-run register function, so we want to only call it once.
            if current_flash_attention_impl() != "FA3":
                activate_flash_attention_impl("FA3")
        else:
            warn_once(
                logger, "FA3 not available (requires SM 9.0+), falling back to FA2. "
            )

    # Based on vLLM's FlashAttentionImpl.forward():
    # https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py
    #
    # @eager_break_during_capture makes this forward a break point for vLLM's
    # breakable cudagraph (VLLM_USE_BREAKABLE_CUDAGRAPH=1, cudagraph_mode=PIECEWISE)
    @eager_break_during_capture
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        assert (
            self.vllm_flash_attn_version is not None
        ), "FlashAttention version not detected."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        # Breakable cudagraph: under VLLM_USE_BREAKABLE_CUDAGRAPH the recorded
        # forward closure pins capture-time args (attn_metadata None, varlen metadata
        # absent). Re-read live per-layer metadata + kv_cache from the forward context
        # (vLLM refreshes them before each replay) BEFORE the None check, else replay
        # short-circuits to output.fill_(0) (zeroed attention). No-op outside capture.
        attn_metadata, _, kv_cache, _ = get_attention_context(layer.layer_name)

        if attn_metadata is None:
            # Profiling / cudagraph dummy-capture run (no real metadata yet).
            return output.fill_(0)

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        assert attn_type not in (
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER,
        ), "Encoder-only attention not supported yet."

        # For decoder and cross-attention, use KV cache as before
        key_cache, value_cache = kv_cache.unbind(1)

        assert not self.kv_cache_dtype.startswith(
            "fp8"
        ), "FP8 KV cache not supported yet."

        assert not attn_metadata.use_cascade, "Cascade not supported yet."

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        assert self.dcp_world_size == 1, "DCP not supported yet."

        if not attn_metadata.causal:
            raise RuntimeError("Non-causal attention not supported yet.")

        # vLLM assigns sliding_window_size = None to (-1, -1) w/ optional causal flag
        # but varlen only encode with sliding window. so we need to convert vllm (-1, -1) to (-1, 0)
        # for proper full causal attention instead of bidirectional
        if self.sliding_window == (-1, -1):
            sliding_window_size = (-1, 0)
        else:
            # by default vLLM sets attention type = DECODER, which will set (W-1, 0)
            sliding_window_size = self.sliding_window

        assert self.alibi_slopes is None, "Alibi slopes not supported yet."

        # FA3 can infer cu_seqlens_k from block_table + seqused_k.
        # FA2 requires cu_seqlens_k to be explicitly set.
        if current_flash_attention_impl() == "FA3":
            cu_seqlens_k = None
        else:
            num_seqs = seqused_k.shape[0]
            cu_seqlens_k = torch.zeros(
                num_seqs + 1, dtype=torch.int32, device=query.device
            )
            cu_seqlens_k[1:] = torch.cumsum(seqused_k, dim=0)
        extra_kwargs: dict[str, Any] = {}

        # TODO(pytorch/pytorch#179760): FA2's auto num_splits heuristic
        # produces NaN intermittently with paged KV (block_table). Force
        # num_splits=1 as a workaround until the root cause is fixed
        # upstream. current_flash_attention_impl() returns None when FA2
        # is the implicit default (SM < 9.0). For FA3, only force
        # num_splits=1 in batch-invariant mode (determinism).

        fa_impl = current_flash_attention_impl()
        if fa_impl in (None, "FA2") or is_in_batch_invariant_mode():
            extra_kwargs["num_splits"] = 1

        if self.enable_gqa:
            extra_kwargs["enable_gqa"] = True

        if self.out_transform is not None:
            extra_kwargs["return_aux"] = AuxRequest(lse=True)

        """
        # Ensure key_cache and value_cache are 4D: [num_blocks, block_size, num_kv_heads, head_dim]
        if block_table is not None and key_cache.ndim == 3:
            num_blocks, block_size, kv_dim = key_cache.shape
            head_dim = query.shape[-1]  # 128
            num_kv_heads = kv_dim // head_dim  # 256 // 128 = 2
            
            key_cache = key_cache.view(num_blocks, block_size, num_kv_heads, head_dim)
            value_cache = value_cache.view(num_blocks, block_size, num_kv_heads, head_dim)
 
        # Insert right above line 245 (result = torch.nn.attention.varlen.varlen_attn_out...):
        print(f"[DEBUG DEVICES]")
        print(f"  output:       {output.device}")
        print(f"  query:        {query.device}")
        print(f"  key_cache:    {key_cache.device}")
        print(f"  value_cache:  {value_cache.device}")
        print(f"  cu_seqlens_q: {cu_seqlens_q.device if cu_seqlens_q is not None else None}")
        print(f"  cu_seqlens_k: {cu_seqlens_k.device if cu_seqlens_k is not None else None}")
        print(f"  block_table:  {block_table.device if block_table is not None else None}")
        print(f"  seqused_k:    {seqused_k.device if seqused_k is not None else None}")


        result = torch.nn.attention.varlen.varlen_attn_out(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            scale=self.scale,
            window_size=sliding_window_size,
            block_table=block_table,
            seqused_k=seqused_k,
            **extra_kwargs,
        )
        if self.out_transform is None:
            return result

        out, lse = result
        out = self.out_transform(out, lse.transpose(0, 1))
        output[:num_actual_tokens].copy_(out)
        return output[:num_actual_tokens]
        
                # Check if running on Intel XPU
        if query.device.type == "xpu":
            # 1. Prepare queries: [total_tokens, H_q, D] ->
            q = query[:num_actual_tokens].transpose(0, 1).unsqueeze(0)

            # 2. Extract active prompt / decoded tokens from key_cache and value_cache
            # key_cache shape: [num_blocks, block_size, H_kv, D]
            #if key_cache.ndim == 4:
            #    k = key_cache.view(-1, key_cache.shape, key_cache.shape).transpose(0, 1).unsqueeze(0)
            #    v = value_cache.view(-1, value_cache.shape, value_cache.shape).transpose(0, 1).unsqueeze(0)
            if key_cache.ndim == 4:
                # key_cache.shape is [num_blocks, block_size, num_kv_heads, head_dim]
                # Flatten the first two dimensions (blocks & block_size) into a single sequence token dimension
                num_blocks, block_size, num_kv_heads, head_dim = key_cache.shape
                k = key_cache.view(-1, num_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
                v = value_cache.view(-1, num_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
 
            else:
                k = key_cache.transpose(0, 1).unsqueeze(0)
                v = value_cache.transpose(0, 1).unsqueeze(0)

            # Slice keys/values to match query token sequence length
            #seq_len = q.shape
            #k = k[:, :, :seq_len, :]
            #v = v[:, :, :seq_len, :]
            seq_len = q.shape[2]
            k = k[:, :, :seq_len, :]
            v = v[:, :, :seq_len, :]



            # 3. Compute attention via native PyTorch SDPA (fully supported on Intel XPU)
            #attn_out = torch.nn.functional.scaled_dot_product_attention(
            #    q, k, v, is_causal=self.is_causal, scale=self.scale
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal, scale=self.scale
            )
            # Reshape back to [total_tokens, H_q, D]
            attn_out = attn_out.squeeze(0).transpose(0, 1).contiguous()
            output[:num_actual_tokens].copy_(attn_out)

            if self.out_transform is None:
                return output[:num_actual_tokens]

            # Dummy LSE tensor in case out_transform is configured
            lse = torch.zeros((q.shape, seq_len), dtype=torch.float32, device=query.device)
            out = self.out_transform(output[:num_actual_tokens], lse.transpose(0, 1))
            output[:num_actual_tokens].copy_(out)
            return output[:num_actual_tokens]

        else:
            # Original CUDA path
            result = torch.nn.attention.varlen.varlen_attn_out(
                output[:num_actual_tokens],
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                scale=self.scale,
                window_size=sliding_window_size,
                block_table=block_table,
                seqused_k=seqused_k,
                **extra_kwargs,
            )
            if self.out_transform is None:
                return result

            out, lse = result
            out = self.out_transform(out, lse.transpose(0, 1))
            output[:num_actual_tokens].copy_(out)
            return output[:num_actual_tokens]
        """
        # Ensure key_cache and value_cache are 4D: [num_blocks, block_size, num_kv_heads, head_dim]
        if block_table is not None and key_cache.ndim == 3:
            num_blocks, block_size, kv_dim = key_cache.shape
            head_dim = query.shape[-1]  # 128
            num_kv_heads = kv_dim // head_dim  # 256 // 128 = 2

            key_cache = key_cache.view(num_blocks, block_size, num_kv_heads, head_dim)
            value_cache = value_cache.view(num_blocks, block_size, num_kv_heads, head_dim)

        # Check if running on Intel XPU
        if query.device.type == "xpu":
            # Qwen decoders use causal attention
            is_causal_xpu = True

            # 1. Prepare queries: [total_tokens, H_q, D] -> [1, H_q, total_tokens, D]
            q = query[:num_actual_tokens].transpose(0, 1).unsqueeze(0)

            # 2. Extract and format active keys/values from cache
            if key_cache.ndim == 4:
                num_blocks, block_size, num_kv_heads, head_dim = key_cache.shape
                k = key_cache.view(-1, num_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
                v = value_cache.view(-1, num_kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
            else:
                k = key_cache.transpose(0, 1).unsqueeze(0)
                v = value_cache.transpose(0, 1).unsqueeze(0)

            # Slice keys/values sequence dimension to match query token sequence length
            #seq_len = q.shape[2]
            #k = k[:, :, :seq_len, :]
            #v = v[:, :, :seq_len, :]

            # 3. Compute attention via native PyTorch SDPA (highly optimized on Intel XPU)
            #attn_out = torch.nn.functional.scaled_dot_product_attention(
            #    q, k, v, is_causal=is_causal_xpu, scale=self.scale
            #)
            # Slice keys/values sequence dimension to match query token sequence length
            seq_len = int(q.shape[2])
            k = k[:, :, :seq_len, :]
            v = v[:, :, :seq_len, :]

            # Align KV heads with Query heads for Grouped Query Attention (GQA)
            num_q_heads = int(q.shape[1])
            num_kv_heads = int(k.shape[1])
            if num_q_heads != num_kv_heads:
                repeat_factor = num_q_heads // num_kv_heads
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)

            # 3. Compute attention via native PyTorch SDPA (highly optimized on Intel XPU)
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=is_causal_xpu, scale=self.scale
            )
 
            
            # Reshape back to [total_tokens, H_q, D]
            attn_out = attn_out.squeeze(0).transpose(0, 1).contiguous()
            output[:num_actual_tokens].copy_(attn_out)

            if self.out_transform is None:
                return output[:num_actual_tokens]

            # Dummy LSE tensor in case out_transform is configured
            lse = torch.zeros((q.shape[1], seq_len), dtype=torch.float32, device=query.device)
            out = self.out_transform(output[:num_actual_tokens], lse.transpose(0, 1))
            output[:num_actual_tokens].copy_(out)
            return output[:num_actual_tokens]

        else:
            # Original CUDA path
            result = torch.nn.attention.varlen.varlen_attn_out(
                output[:num_actual_tokens],
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                scale=self.scale,
                window_size=sliding_window_size,
                block_table=block_table,
                seqused_k=seqused_k,
                **extra_kwargs,
            )
            if self.out_transform is None:
                return result

            out, lse = result
            out = self.out_transform(out, lse.transpose(0, 1))
            output[:num_actual_tokens].copy_(out)
            return output[:num_actual_tokens]


class VLLMAttentionWrapper(Module):
    """Adapter from TorchTitan tensor layout to ``vllm.Attention``.

    vLLM's ``Attention`` layer manages KV-cache and paged attention internally,
    but expects flattened ``(num_tokens, num_heads, head_dim)`` inputs.

    Receives ``(bs, seq, heads, dim)`` layout from GQAttention.

    Used as ``inner_attention`` in GQAttention via Config-based construction.
    """

    # vLLM requires a unique prefix per Attention layer for
    # static_forward_context registration.
    # TODO: Pass layer_id through the build chain instead of using a
    # global counter. The counter breaks with pipeline parallelism
    # where layers are built on different ranks.
    _layer_counter: itertools.count = itertools.count()

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        num_heads: int
        num_kv_heads: int
        head_dim: int
        scale: float | None = None
        sliding_window_size: int | None = None
        """Causal sliding-window size (``None`` => full attention)."""

    def __init__(self, config: Config) -> None:
        super().__init__()

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        tp_degree = vllm_config.parallel_config.tensor_parallel_size

        num_heads = config.num_heads
        num_kv_heads = config.num_kv_heads

        if num_kv_heads < tp_degree:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be >= "
                f"tensor_parallel_size ({tp_degree})"
            )
        if num_kv_heads % tp_degree != 0:
            raise ValueError(
                f"num_kv_heads ({num_kv_heads}) must be divisible by "
                f"tensor_parallel_size ({tp_degree})"
            )
        if num_heads % tp_degree != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"tensor_parallel_size ({tp_degree})"
            )

        num_heads = num_heads // tp_degree
        num_kv_heads = num_kv_heads // tp_degree
        head_dim = config.head_dim
        scale = config.scale if config.scale is not None else head_dim**-0.5

        self.hidden_size = config.hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale

        cache_config = (
            vllm_config.cache_config if hasattr(vllm_config, "cache_config") else None
        )

        # TODO: This need to be compatible with Pipeline Parallelism
        layer_id = next(VLLMAttentionWrapper._layer_counter)
        self.vllm_attn = Attention(
            num_heads=num_heads,
            head_size=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            cache_config=cache_config,
            quant_config=None,
            per_layer_sliding_window=config.sliding_window_size,
            prefix=f"model.layers.{layer_id}.attention.inner_attention",
        )

    def forward(
        self,
        q_BLNH: torch.Tensor,
        k_BLNH: torch.Tensor,
        v_BLNH: torch.Tensor,
        *,
        attention_masks: AttentionMasksType | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run vLLM paged attention on local (non-DTensor) tensors.

        Args:
            q_BLNH: ``(batch, seq_len, num_heads, head_dim)``
            k_BLNH: ``(batch, seq_len, num_kv_heads, head_dim)``
            v_BLNH: ``(batch, seq_len, num_kv_heads, head_dim)``

        Returns:
            ``(batch, seq_len, num_heads * head_dim)`` -- ready for
            ``output.view(bs, seqlen, -1)`` in GQAttention.forward
        """
        if attention_masks is not None:
            raise ValueError(
                "VLLMAttentionWrapper does not support attention_masks; vLLM "
                "manages causal masking and the KV-cache internally."
            )

        batch_size, seq_len, _, head_dim = q_BLNH.shape

        # vllm attention expects (bs*seqlen, n_heads, head_dim) == (T, N, H)
        # (bs, seq, heads, dim) is contiguous, so reshape is zero-copy
        q_TNH = q_BLNH.reshape(batch_size * seq_len, -1, head_dim)
        k_TNH = k_BLNH.reshape(batch_size * seq_len, -1, head_dim)
        v_TNH = v_BLNH.reshape(batch_size * seq_len, -1, head_dim)

        out_TD = self.vllm_attn(q_TNH, k_TNH, v_TNH)

        # vLLM's flash attention backend may pad the token count (e.g.
        # round up to an even number), which introduces a new symbolic
        # shape under torch.compile.  Narrow to trim this padding.
        out_TD = out_TD.narrow(0, 0, batch_size * seq_len)

        # Reshape back to the (B, L, N, H) format expected by GQAttention.forward()
        out_BLNH = out_TD.view(batch_size, seq_len, -1, head_dim)

        return out_BLNH
