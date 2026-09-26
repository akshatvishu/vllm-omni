# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Unit tests for TeaCache extractor functions.

This module provides a generic testing framework for model-specific extractor functions
used by TeaCache. Each model's extractor can be tested by:
1. Creating a fixture that returns model module
2. Creating a fixture that returns sample inputs for that model
3. Creating a test class that inherits from BaseExtractorTest
4. Implementing any model-specific test methods

Currently implemented:
- TestMiniMaxH3Extractor: MiniMaxH3DiTModel extractor
"""

import math
from abc import ABC, abstractmethod
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.cache.teacache.extractors import extract_minimax_h3_context

pytestmark = [pytest.mark.core_model]


@pytest.fixture(scope="function", autouse=True)
def setup_tp_group():
    """Set up TP group for each test function"""
    with patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=1):
        with patch("vllm.distributed.parallel_state.get_tp_group") as mock_get_tp_group:
            mock_tp_group = MagicMock()
            mock_tp_group.world_size = 1
            mock_get_tp_group.return_value = mock_tp_group
            yield


class BaseExtractorTest(ABC):
    """Base class for testing TeaCache extractors.

    Subclasses should implement:
    - get_extractor(): Return extractor function
    - get_module(): Return model module
    - get_sample_inputs(): Return sample inputs for model
    """

    @abstractmethod
    def get_extractor(self):
        """Return extractor function to test."""
        pass

    @abstractmethod
    def get_module(self, module, /):
        """Return model module instance."""
        pass

    @abstractmethod
    def get_sample_inputs(self, sample_inputs, /):
        """Return sample inputs for model."""
        pass


class _MiniMaxH3FakeLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        *,
        bias=False,
        params_dtype=None,
        quant_config=None,
        prefix="",
        total_num_heads=None,
        total_num_kv_heads=None,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        self.prefix = prefix
        self.quant_config = quant_config
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=params_dtype or torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=params_dtype or torch.float32))
        else:
            self.register_parameter("bias", None)
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_kv_heads

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias), None


class _MiniMaxH3FakeMergedLinear(_MiniMaxH3FakeLinear):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(in_features, sum(out_features), **kwargs)


class _MiniMaxH3FakeQKVLinear(_MiniMaxH3FakeLinear):
    def __init__(
        self,
        *,
        hidden_size,
        head_size,
        total_num_heads,
        total_num_kv_heads,
        **kwargs,
    ):
        qkv_features = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(
            hidden_size,
            qkv_features,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            **kwargs,
        )


class _MiniMaxH3FakeAttention(nn.Module):
    class FakeBackend:
        supports_prefix_kv_slicing = False

        # Mirror AttentionBackend.supports_packed_mask_free: H3's packed
        # attention path calls attn_backend.supports_packed_mask_free() to decide
        # whether to build the padding mask. Return False so the fake exercises
        # the masked path (the abstract backend default).
        @classmethod
        def supports_packed_mask_free(cls) -> bool:
            return False

    def __init__(self, **kwargs):
        del kwargs
        super().__init__()
        self.attn_backend = self.FakeBackend
        self.use_ring = False

    def forward(self, q, k, v, metadata=None):
        del k, v, metadata
        return q


def _minimax_h3_small_od_config():
    arch = {
        "num_layers": 1,
        "token_refiner_num_layers": 1,
        "hidden_size": 8,
        "num_attention_heads": 1,
        "attention_head_dim": 8,
        "ffn_hidden_size": 16,
        "latents_dim": 2,
        "audio_latents_dim": 2,
        "patch_size": (1, 2, 2),
        "text_dim": 6,
        "timestep_input_dim": 4,
        "time_embed_hidden_size": 8,
        "time_embed_dim": 4,
        "adaln_out_features": 18 * 8,
        "final_adaln_out_features": 2 * 8,
        "rope_inv_freq_len": 1,
    }
    return SimpleNamespace(
        tf_model_config=arch,
        parallel_config=SimpleNamespace(ulysses_degree=1),
    )


def _minimax_h3_sample_kwargs(*, seq_len: int = 8, device: torch.device | str = "cpu") -> dict:
    video_row_dim = 2 * 1 * 2 * 2
    audio_row_dim = 2
    text_len = 2
    img_rows = 2

    img_pos = torch.tensor([2, 3], dtype=torch.long, device=device)
    audio_pos = torch.tensor([4, 5], dtype=torch.long, device=device)
    text_pos = torch.tensor([0, 1], dtype=torch.long, device=device)
    infer_out_pos = img_pos.clone()

    token_tags = torch.full((seq_len,), -1, dtype=torch.long, device=device)
    token_tags[text_pos] = 1
    token_tags[img_pos] = 0
    token_tags[audio_pos] = 2

    inverse_indices = torch.zeros(seq_len, dtype=torch.long, device=device)

    return {
        "x": torch.randn(1, seq_len, video_row_dim, device=device),
        "audio_x": torch.randn(1, seq_len, audio_row_dim, device=device),
        "img_position_ids": torch.zeros(1, seq_len, 3, dtype=torch.float64, device=device),
        "unique_timesteps": torch.tensor([0.5], dtype=torch.float32, device=device),
        "inverse_indices": inverse_indices,
        "update_mask": torch.ones(img_rows, dtype=torch.bool, device=device),
        "token_tags": token_tags,
        "prompt_embeds": torch.randn(text_len, 6, device=device),
        "img_pos_info": {"position_ids": img_pos},
        "audio_pos_info": {"position_ids": audio_pos},
        "text_pos_info": {"position_ids": text_pos},
        "img_pos_for_infer_output_info": {"position_ids": infer_out_pos},
        "packed_seq_params": {
            "cu_seqlens_q": torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            "max_seqlen_q": seq_len,
        },
        "refiner_packed_seq_params": {
            "cu_seqlens_q": torch.tensor([0, text_len], dtype=torch.int32, device=device),
            "max_seqlen_q": text_len,
        },
    }


@pytest.mark.cpu
class TestMiniMaxH3Extractor(BaseExtractorTest):
    """Test extract_minimax_h3_context function."""

    def get_extractor(self):
        return extract_minimax_h3_context

    @pytest.fixture
    def minimax_h3_module(self, monkeypatch):
        """Create a minimal MiniMaxH3DiTModel with fake parallel linears for CPU tests."""
        from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3

        monkeypatch.setattr(h3, "ColumnParallelLinear", _MiniMaxH3FakeLinear)
        monkeypatch.setattr(h3, "MergedColumnParallelLinear", _MiniMaxH3FakeMergedLinear)
        monkeypatch.setattr(h3, "QKVParallelLinear", _MiniMaxH3FakeQKVLinear)
        monkeypatch.setattr(h3, "RowParallelLinear", _MiniMaxH3FakeLinear)
        monkeypatch.setattr(h3, "Attention", _MiniMaxH3FakeAttention)
        monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)

        model = h3.MiniMaxH3DiTModel(_minimax_h3_small_od_config(), quant_config=None)
        # Normally loaded from the checkpoint; torch.empty() can contain NaNs.
        # With one frequency per axis, the RoPE inverse frequency is 1.
        model.rope.inv_freq.fill_(1.0)
        for submodule in model.modules():
            if isinstance(submodule, h3.MiniMaxH3Attention):
                submodule.rope._forward_method = submodule.rope.forward_native
        model.eval()
        return model

    def get_module(self, minimax_h3_module):
        return minimax_h3_module

    @pytest.fixture
    def sample_inputs(self):
        """Create sample packed forward kwargs for MiniMaxH3DiTModel."""
        return _minimax_h3_sample_kwargs(seq_len=8)

    def get_sample_inputs(self, sample_inputs):
        return sample_inputs

    def test_modulated_input_shape(self, minimax_h3_module, sample_inputs):
        """Test that modulated_input matches packed sequence shape [T, H]."""
        context = extract_minimax_h3_context(minimax_h3_module, **sample_inputs)

        seq_len = sample_inputs["x"].shape[1]
        assert context.modulated_input.shape == (seq_len, minimax_h3_module.hidden_size)

    def test_run_transformer_blocks_callable(self, minimax_h3_module, sample_inputs):
        """Test that run_transformer_blocks is callable."""
        context = extract_minimax_h3_context(minimax_h3_module, **sample_inputs)
        assert callable(context.run_transformer_blocks)

    def test_postprocess_callable(self, minimax_h3_module, sample_inputs):
        """Test that postprocess is callable."""
        context = extract_minimax_h3_context(minimax_h3_module, **sample_inputs)
        assert callable(context.postprocess)

    def test_postprocess_output_shape(self, minimax_h3_module, sample_inputs):
        """Test that postprocess returns video/audio logits for infer rows."""
        context = extract_minimax_h3_context(minimax_h3_module, **sample_inputs)
        (hidden,) = context.run_transformer_blocks()
        video_logits, audio_logits = context.postprocess(hidden)

        num_video_rows = sample_inputs["img_pos_info"]["position_ids"].shape[0]
        num_audio_rows = sample_inputs["audio_pos_info"]["position_ids"].shape[0]
        assert video_logits.shape[0] == num_video_rows
        assert audio_logits.shape[0] == num_audio_rows

    def test_extractor_matches_native_forward(self, minimax_h3_module, sample_inputs):
        """The uncached extractor path must preserve the model's forward result."""
        expected_video, expected_audio = minimax_h3_module(**sample_inputs)

        context = extract_minimax_h3_context(minimax_h3_module, **sample_inputs)
        (hidden,) = context.run_transformer_blocks()
        actual_video, actual_audio = context.postprocess(hidden)

        torch.testing.assert_close(actual_video, expected_video)
        torch.testing.assert_close(actual_audio, expected_audio)

    def test_extractor_uses_prepared_rope_table(
        self,
        minimax_h3_module,
        sample_inputs,
        monkeypatch,
    ):
        """TeaCache must consume the denoise-loop RoPE table without rebuilding it."""
        rope_table = minimax_h3_module.prepare_rope_table(
            sample_inputs["img_position_ids"],
            seq_len=sample_inputs["x"].shape[1],
        )

        def unexpected_rope_build(*args, **kwargs):
            raise AssertionError("TeaCache extractor rebuilt the prepared RoPE table")

        monkeypatch.setattr(minimax_h3_module, "prepare_rope_table", unexpected_rope_build)
        monkeypatch.setattr(minimax_h3_module.rope, "forward", unexpected_rope_build)
        context = extract_minimax_h3_context(
            minimax_h3_module,
            **{**sample_inputs, "rope_table": rope_table},
        )
        context.run_transformer_blocks()

    def test_run_transformer_blocks_forwards_packed_layout(
        self,
        minimax_h3_module,
        sample_inputs,
        monkeypatch,
    ):
        """The extractor block call must mirror the native packed contract."""
        from vllm_omni.diffusion.attention.backends.abstract import VideoTokenLayout

        video_layout = VideoTokenLayout(prefix_len=2, latent_grid=(1, 1, 2))
        inputs = {**sample_inputs, "video_token_layout": video_layout}
        block = minimax_h3_module.blocks[0]
        original_forward = block.forward
        received: list[tuple[int, VideoTokenLayout | None]] = []

        def capture_forward(*args, **kwargs):
            received.append((kwargs["packed_total"], kwargs["video_layout"]))
            return original_forward(*args, **kwargs)

        monkeypatch.setattr(block, "forward", capture_forward)
        context = extract_minimax_h3_context(minimax_h3_module, **inputs)
        context.run_transformer_blocks()

        assert received == [(sample_inputs["x"].shape[1], video_layout)]

    def test_teacache_reuses_h3_block_residual(self, minimax_h3_module, sample_inputs, monkeypatch):
        """A guaranteed cache hit must skip the H3 transformer block."""
        from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
        from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook

        block = minimax_h3_module.blocks[0]
        original_forward = block.forward
        block_calls = 0

        def counted_forward(*args, **kwargs):
            nonlocal block_calls
            block_calls += 1
            return original_forward(*args, **kwargs)

        monkeypatch.setattr(block, "forward", counted_forward)
        hook = TeaCacheHook(
            TeaCacheConfig(
                transformer_type="MiniMaxH3DiTModel",
                rel_l1_thresh=0.3,
            )
        )
        hook.initialize_hook(minimax_h3_module)

        hook.new_forward(minimax_h3_module, **sample_inputs)
        video_logits, audio_logits = hook.new_forward(minimax_h3_module, **sample_inputs)

        assert block_calls == 1
        assert video_logits.shape[0] == sample_inputs["img_pos_info"]["position_ids"].shape[0]
        assert audio_logits.shape[0] == sample_inputs["audio_pos_info"]["position_ids"].shape[0]

    @pytest.mark.sp
    def test_teacache_strict_sp_keeps_cache_state_rank_local(
        self,
        minimax_h3_module,
        sample_inputs,
        monkeypatch,
    ):
        """Strict SP must not mix local TeaCache state with gathered block rows."""
        from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
        from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook
        from vllm_omni.diffusion.distributed import parallel_state
        from vllm_omni.diffusion.layers import indexed_modulation

        seq_len = sample_inputs["x"].shape[1]
        local_len = seq_len // 2
        hidden_size = minimax_h3_module.hidden_size
        video_width = minimax_h3_module.arch.latents_dim * math.prod(minimax_h3_module.arch.patch_size)
        audio_width = minimax_h3_module.arch.audio_latents_dim
        captured_modulation_indices: list[torch.Tensor] = []

        class _StrictSPPrepare(nn.Module):
            def __init__(self):
                super().__init__()
                self.global_indices: list[torch.Tensor] = []

            def forward(self, hidden, rope_table, combined_indices):
                assert hidden.shape[0] == local_len
                assert rope_table.shape[0] == local_len
                self.global_indices.append(combined_indices.clone())
                return hidden, rope_table, combined_indices.narrow(0, 0, local_len)

        class _StrictSPGather(nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs: list[torch.Tensor] = []
                self.outputs: list[torch.Tensor] = []

            def forward(self, tensor):
                self.inputs.append(tensor.clone())
                if tensor.shape[-1] == hidden_size:
                    raise AssertionError("TeaCache must not gather hidden states in strict SP")
                assert tensor.shape == (local_len, video_width + audio_width)
                # Model the all-gather of compact logits from every SP rank.
                # postprocess() indexes the resulting global layout with
                # infer_out_pos/audio_pos, including rows owned by another rank.
                gathered = tensor.repeat(seq_len // local_len, 1)
                assert gathered.shape == (seq_len, video_width + audio_width)
                self.outputs.append(gathered)
                return gathered

        class _StrictSPGroup:
            def __init__(self):
                super().__init__()
                self.local_decisions: list[int] = []
                self.ops = []

            def all_reduce(self, decision, op):
                self.local_decisions.append(int(decision.item()))
                self.ops.append(op)
                # Simulate another rank requiring the second step to execute
                # the collective block path even though this rank would cache.
                if len(self.local_decisions) == 2:
                    decision.fill_(1)
                return decision

        class _StrictSPBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.block_indices: list[torch.Tensor] = []

            def adaln_proj(self, t_emb):
                del t_emb
                zeros = torch.zeros(3, hidden_size, dtype=torch.float32)
                return zeros, zeros

            def norm1(self, hidden):
                return hidden.clone()

            def forward(self, hidden, *, combined_indices, rope_table, **kwargs):
                del kwargs
                assert hidden.shape[0] == local_len
                assert rope_table.shape[0] == local_len
                self.block_indices.append(combined_indices.clone())
                return hidden + 1

        class _StrictSPFinalLayer(nn.Module):
            def forward(self, hidden, *, t_emb, inverse_indices):
                del t_emb, inverse_indices
                assert hidden.shape == (local_len, hidden_size)
                return hidden[:, :video_width].float(), hidden[:, :audio_width].float()

        local_prepare = _StrictSPPrepare()
        gather = _StrictSPGather()
        sp_group = _StrictSPGroup()
        block = _StrictSPBlock()
        minimax_h3_module.local_sp_prepare = local_prepare
        minimax_h3_module.sp_gather = gather
        minimax_h3_module.blocks = nn.ModuleList([block])
        minimax_h3_module.final_layer = _StrictSPFinalLayer()
        monkeypatch.setattr(minimax_h3_module, "_rope_local_span", lambda _seq_len: (0, local_len))
        monkeypatch.setattr(
            minimax_h3_module,
            "_embed",
            lambda **kwargs: (
                torch.arange(local_len * hidden_size, dtype=torch.float32).reshape(local_len, hidden_size),
                torch.zeros(3, hidden_size, dtype=torch.float32),
            ),
        )
        monkeypatch.setattr(
            minimax_h3_module,
            "prepare_rope_table",
            lambda *args, **kwargs: torch.zeros(local_len, 6, dtype=torch.float32),
        )
        monkeypatch.setattr(parallel_state, "get_sp_group", lambda: sp_group)

        original_indexed_scale_shift = indexed_modulation.indexed_scale_shift_

        def capture_indexed_scale_shift(hidden, shift, scale, indices):
            captured_modulation_indices.append(indices.clone())
            return original_indexed_scale_shift(hidden, shift, scale, indices)

        monkeypatch.setattr(indexed_modulation, "indexed_scale_shift_", capture_indexed_scale_shift)

        local_img_pos = torch.tensor([0, 1])
        local_audio_pos = torch.tensor([2, 3])
        token_tags = torch.full((seq_len,), -1, dtype=torch.long)
        token_tags[local_img_pos] = 0
        token_tags[local_audio_pos] = 2
        inverse_indices = torch.zeros(seq_len, dtype=torch.long)
        inputs = {
            **sample_inputs,
            "inverse_indices": inverse_indices,
            "token_tags": token_tags,
            "update_mask": torch.ones(local_img_pos.numel(), dtype=torch.bool),
            "prompt_embeds": torch.empty(0, 6),
            "img_pos_info": {"position_ids": local_img_pos},
            "audio_pos_info": {"position_ids": local_audio_pos},
            "text_pos_info": {"position_ids": torch.empty(0, dtype=torch.long)},
            "img_pos_for_infer_output_info": {"position_ids": local_img_pos},
            "refiner_packed_seq_params": {
                "cu_seqlens_q": torch.tensor([0, 0], dtype=torch.int32),
                "max_seqlen_q": 0,
            },
        }
        expected_local_indices = (inverse_indices * 3 + token_tags.clamp(min=0))[:local_len]

        hook = TeaCacheHook(
            TeaCacheConfig(
                transformer_type="MiniMaxH3DiTModel",
                rel_l1_thresh=0.3,
            )
        )
        hook.initialize_hook(minimax_h3_module)

        hook.new_forward(minimax_h3_module, **inputs)
        hook.new_forward(minimax_h3_module, **inputs)
        video_logits, audio_logits = hook.new_forward(minimax_h3_module, **inputs)
        state = hook.state_manager.get_state()

        assert state.previous_residual is not None
        assert state.previous_residual.shape == (local_len, hidden_size)
        assert len(captured_modulation_indices) == 3
        assert all(torch.equal(indices, expected_local_indices) for indices in captured_modulation_indices)
        assert sp_group.local_decisions == [1, 0, 0]
        assert len(sp_group.ops) == 3
        assert all(op == torch.distributed.ReduceOp.MAX for op in sp_group.ops)
        assert len(local_prepare.global_indices) == 2
        assert all(indices.shape == (seq_len,) for indices in local_prepare.global_indices)
        assert len(block.block_indices) == 2
        assert all(torch.equal(indices, expected_local_indices) for indices in block.block_indices)
        assert len(gather.inputs) == 3
        assert video_logits.shape == (local_img_pos.numel(), video_width)
        assert audio_logits.shape == (local_audio_pos.numel(), audio_width)

    @pytest.mark.cpu
    def test_teacache_non_strict_sp_gathers_hidden_once(
        self,
        minimax_h3_module,
        sample_inputs,
        monkeypatch,
    ):
        """The non-strict SP cache path must not gather global hidden rows twice."""
        from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
        from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook

        class _Prepare(nn.Module):
            def forward(self, hidden, rope_table, combined_indices):
                return hidden, rope_table, combined_indices

        class _Gather(nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs: list[torch.Tensor] = []

            def forward(self, tensor):
                self.inputs.append(tensor.clone())
                return tensor

        seq_len = sample_inputs["x"].shape[1]
        gather = _Gather()
        minimax_h3_module.sp_prepare = _Prepare()
        minimax_h3_module.sp_gather = gather
        monkeypatch.setattr(minimax_h3_module, "_rope_local_span", lambda _seq_len: (0, seq_len))

        hook = TeaCacheHook(
            TeaCacheConfig(
                transformer_type="MiniMaxH3DiTModel",
                rel_l1_thresh=0.3,
            )
        )
        hook.initialize_hook(minimax_h3_module)
        hook.new_forward(minimax_h3_module, **sample_inputs)

        assert len(gather.inputs) == 1
        assert gather.inputs[0].shape == (seq_len, minimax_h3_module.hidden_size)

    def test_invalid_module_raises_error(self):
        """Test that invalid module without blocks raises ValueError."""
        invalid_module = Mock()
        invalid_module.blocks = []

        with pytest.raises(ValueError, match="Module must have blocks"):
            extract_minimax_h3_context(invalid_module, **_minimax_h3_sample_kwargs())
