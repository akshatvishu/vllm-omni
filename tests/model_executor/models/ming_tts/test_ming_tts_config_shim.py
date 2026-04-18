# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from transformers import AutoConfig

from vllm_omni.engine.arg_utils import _register_omni_hf_configs
from vllm_omni.model_executor.models.ming_tts.configuration_ming_dense import MingDenseConfig


def test_ming_dense_autoconfig_registration_uses_local_config(tmp_path):
    _register_omni_hf_configs()
    model_dir = tmp_path / "ming"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        """
{
  "model_type": "dense",
  "auto_map": {"AutoConfig": "configuration_bailingmm.BailingMMConfig"},
  "llm_config": {
    "model_type": "qwen2",
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "vocab_size": 151936
  },
  "audio_tokenizer_config": {
    "sample_rate": 44100,
    "patch_size": 4,
    "enc_kwargs": {
      "latent_dim": 64,
      "input_dim": 882,
      "hop_size": 882,
      "backbone": {"attn_implementation": "flash_attention_2"}
    },
    "dec_kwargs": {
      "latent_dim": 64,
      "output_dim": 882,
      "backbone": {"_attn_implementation": "flash_attention_2"}
    }
  }
}
""".strip()
    )

    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=False, local_files_only=True)

    assert isinstance(cfg, MingDenseConfig)
    assert cfg.get_text_config().num_attention_heads == 14
    assert cfg.audio_tokenizer_config.sample_rate == 44100
    assert cfg.audio_tokenizer_config.patch_size == 4
