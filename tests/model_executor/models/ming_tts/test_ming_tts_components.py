# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.ming_tts.aggregator import Aggregator
from vllm_omni.model_executor.models.ming_tts.audio_tokenizer.configuration_audio_vae import AudioVAEconfig
from vllm_omni.model_executor.models.ming_tts.audio_tokenizer.istft import ISTFT, ISTFTHead
from vllm_omni.model_executor.models.ming_tts.audio_tokenizer.modeling_audio_vae import AudioVAE
from vllm_omni.model_executor.models.ming_tts.audio_tokenizer.vae_modules import StreamingLinearUpsample
from vllm_omni.model_executor.models.ming_tts.flowloss_head import FlowLoss
from vllm_omni.model_executor.models.ming_tts.fm.cfm import CFM, Solver, get_epss_timesteps
from vllm_omni.model_executor.models.ming_tts.fm.dit import (
    CondEmbedder,
    DiT,
    SinusPositionEmbedding,
    TimestepEmbedder,
)
from vllm_omni.model_executor.models.ming_tts.fm.modules import Attention, DiTBlock, RMSNorm
from vllm_omni.model_executor.models.ming_tts.ming_tts import (
    _coerce_prompt_latents,
    _find_audio_placeholder_positions,
    _initial_history,
)
from vllm_omni.model_executor.models.ming_tts.ming_tts_audio_vae import _coerce_finished, _coerce_latent_chunk
from vllm_omni.model_executor.models.ming_tts.ming_tts_llm import _coerce_latent_history
from vllm_omni.model_executor.stage_input_processors.ming_tts import llm2audio_vae_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tiny_qwen_config(hidden_size=8):
    return {
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "vocab_size": 32,
        "max_position_embeddings": 64,
    }


def _tiny_audio_vae_config():
    return AudioVAEconfig(
        sample_rate=16000,
        patch_size=2,
        enc_kwargs={
            "backbone": _tiny_qwen_config(),
            "input_dim": 4,
            "hop_size": 4,
            "latent_dim": 2,
        },
        dec_kwargs={
            "backbone": _tiny_qwen_config(),
            "output_dim": 4,
            "latent_dim": 2,
        },
        semantic_module_kwargs=None,
    )


class _DummyCFMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x, t, c, latent_history, mask=None):
        del t, c, latent_history
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x

    def forward_with_cfg(self, x, t, c, cfg_scale, latent_history, patch_size):
        del t, c, cfg_scale, latent_history
        cond = x[:, -patch_size:, :] + 1.0
        uncond = x[:, -patch_size:, :]
        return torch.cat([cond, uncond], dim=0)


def test_rmsnorm_preserves_shape_and_dtype():
    norm = RMSNorm(dim=8, eps=1e-6)
    x = torch.randn(2, 3, 8, dtype=torch.float32)

    out = norm(x)

    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_attention_forward_shape_and_mask():
    attn = Attention(dim=8, heads=2, dim_head=4, dropout=0.0)
    x = torch.randn(1, 5, 8)
    mask = torch.tensor([[True, True, True, True, False]])

    out = attn(x, mask=mask)

    assert out.shape == x.shape
    assert torch.allclose(out[:, -1], torch.zeros_like(out[:, -1]))


def test_attention_rejects_bad_mask_shape():
    attn = Attention(dim=8, heads=2, dim_head=4, dropout=0.0)
    x = torch.randn(1, 5, 8)

    with pytest.raises(ValueError, match="Mask shape mismatch"):
        attn(x, mask=torch.ones(1, 4, dtype=torch.bool))


def test_dit_block_forward_shape():
    block = DiTBlock(hidden_size=8, num_heads=2, mlp_ratio=2.0, dropout=0.0)
    x = torch.randn(1, 5, 8)
    mask = torch.ones(1, 5, dtype=torch.bool)

    out = block(x, mask, rope=None)

    assert out.shape == x.shape


def test_sinus_position_embedding_shape():
    embed = SinusPositionEmbedding(dim=8)
    t = torch.tensor([0.0, 1.0], dtype=torch.float32)

    out = embed(t)

    assert out.shape == (2, 8)


def test_timestep_embedder_distinguishes_steps():
    embedder = TimestepEmbedder(dim=8, freq_embed_dim=8)

    out_a = embedder(torch.tensor([0.0], dtype=torch.float32))
    out_b = embedder(torch.tensor([1.0], dtype=torch.float32))

    assert out_a.shape == (1, 8)
    assert not torch.allclose(out_a, out_b)


def test_cond_embedder_rejects_bad_rank():
    embedder = CondEmbedder(input_feature_size=4, hidden_size=8, dropout_prob=0.0)

    with pytest.raises(ValueError, match="rank-3"):
        embedder(torch.randn(1, 4), train=False)


def test_cond_drop_preserves_conditioning_dtype():
    embedder = CondEmbedder(input_feature_size=4, hidden_size=8, dropout_prob=1.0)
    llm_cond = torch.randn(1, 1, 4, dtype=torch.float16)

    out = embedder.cond_drop(llm_cond)

    assert out.dtype == llm_cond.dtype


def test_dit_forward_shape():
    model = DiT(
        in_channels=2,
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        llm_cond_dim=4,
        cfg_dropout_prob=0.0,
    )
    x = torch.randn(1, 2, 2)
    latent_history = torch.randn(1, 4, 2)
    c = torch.randn(1, 1, 4)
    mask = torch.ones(1, 2, dtype=torch.bool)

    out = model(x=x, t=torch.tensor([0.5]), c=c, latent_history=latent_history, mask=mask)

    assert out.shape == (1, 7, 2)


def test_dit_forward_with_cfg_preserves_conditioning_dtype(monkeypatch):
    model = DiT(
        in_channels=2,
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        llm_cond_dim=4,
        cfg_dropout_prob=0.0,
    )
    seen = {}

    def _fake_forward(x, t, c, latent_history, mask=None):
        del x, t, latent_history, mask
        seen["dtype"] = c.dtype
        return torch.zeros((c.shape[0], 7, 2), dtype=torch.float32)

    monkeypatch.setattr(model, "forward", _fake_forward)
    x = torch.randn(1, 2, 2, dtype=torch.float16)
    latent_history = torch.randn(1, 4, 2, dtype=torch.float16)
    c = torch.randn(1, 1, 4, dtype=torch.float16)

    model.forward_with_cfg(
        x=x,
        t=torch.tensor([0.5], dtype=torch.float16),
        c=c,
        cfg_scale=2.0,
        latent_history=latent_history,
        patch_size=2,
    )

    assert seen["dtype"] == c.dtype


def test_aggregator_forward_shape():
    agg = Aggregator(
        in_channels=2,
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        llm_input_dim=4,
    )
    x = torch.randn(2, 3, 2)
    mask = torch.ones(2, 3, dtype=torch.bool)

    out = agg(x, mask=mask)

    assert out.shape == (2, 1, 4)


def test_get_epss_timesteps_predefined_and_fallback():
    predefined = get_epss_timesteps(10, device=torch.device("cpu"), dtype=torch.float32)
    fallback = get_epss_timesteps(9, device=torch.device("cpu"), dtype=torch.float32)

    assert predefined.shape == (11,)
    assert torch.allclose(predefined[-1], torch.tensor(1.0))
    assert fallback.shape == (10,)
    assert torch.allclose(fallback, torch.linspace(0, 1, 10))


def test_solver_integrate_zero_function_is_stable():
    y0 = torch.ones(1, 2, 2)
    solver = Solver(lambda t, y: torch.zeros_like(y), y0=y0, sigma=0.0, temperature=0.0)
    t = torch.linspace(0, 1, 4)

    out = solver.integrate(t)

    assert out.shape == (4, 1, 2, 2)
    assert torch.allclose(out[0], y0)
    assert torch.allclose(out[-1], y0)


def test_cfm_forward_returns_scalar_loss():
    torch.manual_seed(0)
    cfm = CFM(model=_DummyCFMModel())
    cond = torch.randn(1, 1, 4)
    target = torch.randn(1, 2, 2)
    latent_history = torch.randn(1, 4, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)

    loss = cfm(cond=cond, target=target, latent_history=latent_history, mask=mask, patch_size=2)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_cfm_sample_returns_sample_and_trajectory():
    torch.manual_seed(0)
    cfm = CFM(model=_DummyCFMModel())
    noise = torch.randn(1, 2, 2)
    cond = torch.randn(1, 1, 4)
    latent_history = torch.randn(1, 4, 2)

    out, trajectory = cfm.sample(noise=noise, c=cond, latent_history=latent_history, steps=4, patch_size=2)

    assert out.shape == (1, 2, 2)
    assert trajectory.shape == (5, 1, 2, 2)


def test_cfm_sample_rejects_low_cfg_scale():
    cfm = CFM(model=_DummyCFMModel())
    noise = torch.randn(1, 2, 2)
    cond = torch.randn(1, 1, 4)
    latent_history = torch.randn(1, 4, 2)

    out, trajectory = cfm.sample(
        noise=noise,
        c=cond,
        latent_history=latent_history,
        cfg_scale=0.0,
        patch_size=2,
    )

    assert out.shape == (1, 2, 2)
    assert trajectory.ndim == 4


def test_flowloss_sample_returns_tensor_shape_and_dtype(monkeypatch):
    flow = FlowLoss(
        z_channels=2,
        llm_cond_dim=4,
        hidden_size=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        cfg_dropout_prob=0.0,
    )

    def _fake_sample(**kwargs):
        noise = kwargs["noise"]
        return noise.transpose(1, 2), torch.zeros(1)

    monkeypatch.setattr(flow.cfm, "sample", _fake_sample)
    z = torch.randn(1, 1, 4, dtype=torch.float32)
    latent_history = torch.randn(1, 4, 2, dtype=torch.float32)

    out = flow.sample(z=z, latent_history=latent_history, patch_size=3)

    assert out.shape == (1, 3, 2)
    assert out.dtype == z.dtype


def test_streaming_linear_upsample_rejects_empty_final_flush():
    upsample = StreamingLinearUpsample(scale_factor=2)

    with pytest.raises(ValueError, match="end-of-stream"):
        upsample(None, state=None, is_last=True)


def test_streaming_linear_upsample_streams_and_flushes():
    upsample = StreamingLinearUpsample(scale_factor=2)
    chunk_a = torch.randn(1, 2, 3)
    chunk_b = torch.randn(1, 2, 3)

    out_a, state = upsample(chunk_a, state=None, is_last=False)
    out_b, state = upsample(chunk_b, state=state, is_last=True)

    assert out_a is None
    assert out_b is not None
    assert out_b.shape[0] == 1
    assert out_b.shape[-1] == 3
    assert state is None


def test_istft_rejects_bad_rank():
    istft = ISTFT(n_fft=16, hop_length=4, win_length=16, padding="same")

    with pytest.raises(ValueError, match="rank-3"):
        istft(torch.randn(1, 9))


def test_istft_head_output_shape():
    head = ISTFTHead(dim=8, n_fft=16, hop_length=4, padding="same")
    x = torch.randn(1, 3, 8)

    audio, spec, audio_buffer, window_buffer = head(x)

    assert audio.shape[0] == 1
    assert audio.shape[1] == 1
    assert spec.shape == (1, 18, 3)
    assert audio_buffer is None
    assert window_buffer is None


def test_audio_vae_encode_and_decode_shapes():
    torch.manual_seed(0)
    vae = AudioVAE(_tiny_audio_vae_config())
    waveform = torch.randn(1, 12)
    waveform_length = torch.tensor([12], dtype=torch.int32)

    latent, frame_num = vae.encode_latent(waveform, waveform_length)
    audio, stream_state, past_key_values = vae.decode(latent, use_cache=False)

    assert latent.ndim == 3
    assert latent.shape[0] == 1
    assert latent.shape[-1] == 2
    assert frame_num.tolist() == [2]
    assert audio.ndim == 3
    assert audio.shape[0] == 1
    assert audio.shape[1] == 1
    assert stream_state == (None, None, None)
    assert past_key_values is None


def test_audio_vae_rejects_invalid_inputs():
    vae = AudioVAE(_tiny_audio_vae_config())

    with pytest.raises(ValueError, match="waveform rank-2"):
        vae.encode_latent(torch.randn(12), torch.tensor([12], dtype=torch.int32))

    with pytest.raises(ValueError, match="Latent dim mismatch"):
        vae.decode(torch.randn(1, 2, 3))


def test_coerce_prompt_latents_supports_frames_and_patch_groups():
    frames = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    patches = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)

    out_frames = _coerce_prompt_latents(frames, patch_size=2, latent_dim=2)
    out_patches = _coerce_prompt_latents(patches, patch_size=2, latent_dim=4)

    assert out_frames["patches"].shape == (2, 2, 2)
    assert out_frames["frames"].shape == (4, 2)
    assert out_patches["patches"].shape == (2, 2, 4)
    assert out_patches["frames"].shape == (4, 4)


def test_initial_history_keeps_tail():
    frames = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    history = _initial_history(
        frames,
        history_size=4,
        latent_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert history.shape == (4, 2)
    assert torch.allclose(history, frames[-4:])


def test_find_audio_placeholder_positions_uses_audio_span():
    cfg = SimpleNamespace(
        audio_dummy_token_id=151705,
        audio_start_token_id=151706,
        audio_end_token_id=151707,
    )
    input_ids = torch.tensor([151705, 1, 151706, 151705, 151705, 151707, 151705], dtype=torch.long)

    out = _find_audio_placeholder_positions(input_ids, cfg)

    assert out.tolist() == [3, 4]


def test_helper_coercions_fail_loudly():
    cfg = SimpleNamespace(history_patch_size=4, latent_dim=2)

    assert _coerce_finished(torch.tensor([1], dtype=torch.bool)) is True
    latent_chunk = _coerce_latent_chunk(
        torch.ones(4, 2),
        device=torch.device("cpu"),
        dtype=torch.float32,
        latent_dim=2,
        patch_size=4,
    )
    assert latent_chunk.shape == (1, 4, 2)

    grouped_chunk = _coerce_latent_chunk(
        torch.ones(2, 4, 2),
        device=torch.device("cpu"),
        dtype=torch.float32,
        latent_dim=2,
        patch_size=4,
    )
    assert grouped_chunk.shape == (2, 4, 2)

    with pytest.raises(RuntimeError, match="latent_history shape mismatch"):
        _coerce_latent_history(torch.ones(3, 2), device=torch.device("cpu"), dtype=torch.float32, cfg=cfg)

    with pytest.raises(ValueError, match="Latent patch size mismatch"):
        _coerce_latent_chunk(
            torch.ones(1, 3, 2),
            device=torch.device("cpu"),
            dtype=torch.float32,
            latent_dim=2,
            patch_size=4,
        )

    with pytest.raises(ValueError, match="Latent dim mismatch"):
        _coerce_latent_chunk(
            torch.ones(4, 3),
            device=torch.device("cpu"),
            dtype=torch.float32,
            latent_dim=2,
            patch_size=4,
        )


def test_ming_async_chunk_rejects_left_context_replay():
    transfer_manager = SimpleNamespace(
        connector=SimpleNamespace(config={"extra": {"latent_chunk_size": 10, "latent_left_context": 1}}),
        put_req_chunk={"req-1": 0},
        request_payload={},
    )
    request = SimpleNamespace(external_req_id="req-1", is_finished=lambda: False)

    with pytest.raises(ValueError, match="latent_left_context replay"):
        llm2audio_vae_async_chunk(
            transfer_manager=transfer_manager,
            pooling_output=None,
            request=request,
            is_finished=False,
        )


def test_coerce_latent_history_casts_to_requested_dtype():
    cfg = SimpleNamespace(history_patch_size=4, latent_dim=2)

    history = _coerce_latent_history(
        torch.ones(1, 4, 2, dtype=torch.float16),
        device=torch.device("cpu"),
        dtype=torch.float32,
        cfg=cfg,
    )

    assert history.dtype == torch.float32
