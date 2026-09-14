# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.config.watermarking import WatermarkConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("key", ["message", "seed", "strength"])
def test_watermark_config_rejects_unused_algorithm_options(key: str) -> None:
    with pytest.raises(ValueError, match=f"unsupported watermark config keys for audio: {key}"):
        WatermarkConfig({"audio": {"algorithm": "audioseal", key: 1}})


@pytest.mark.parametrize("config", [{}, {"audio": {"algorithm": "audioseal"}}])
def test_watermark_config_accepts_supported_options(config: dict) -> None:
    assert WatermarkConfig(config).modality_configs == config
