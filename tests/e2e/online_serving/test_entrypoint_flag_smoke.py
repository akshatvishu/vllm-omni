# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check one observable flag through the complete Omni serving path.

The test starts the production ``vllm serve --omni`` subprocess and sends real
HTTP requests. It proves that ``--max-generated-image-size`` remains available
to the frontend consumer after server startup.

Config propagation for the other audited flags stays in
``tests/entrypoints/test_entrypoint_flags_audit.py``. Direct final config
assertions identify those failures more clearly than repeated GPU generation.
"""

from __future__ import annotations

import base64
import io

import pytest
import requests
import torch
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServer

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="requires one CUDA or ROCm GPU",
    ),
]

MODEL = "tiny-random/Qwen-Image"


def _request(server: OmniServer, *, size: str) -> requests.Response:
    return requests.post(
        f"http://{server.host}:{server.port}/v1/images/generations",
        json={
            "model": server.model,
            "prompt": "A small red cube on a white background.",
            "size": size,
            "n": 1,
            "response_format": "b64_json",
            "num_inference_steps": 1,
            "guidance_scale": 0.0,
            "seed": 1,
        },
        timeout=600,
    )


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
def test_max_generated_image_size_from_cli_reaches_the_live_server():
    """Reject an oversized request and complete a valid request."""
    server_args = [
        "--enforce-eager",
        "--max-num-seqs",
        "1",
        "--max-generated-image-size",
        str(256 * 256),
        "--stage-init-timeout",
        "300",
        "--init-timeout",
        "900",
    ]

    with OmniServer(MODEL, server_args, use_omni=True) as server:
        rejected = _request(server, size="512x512")
        assert rejected.status_code == 400
        assert "exceeds the maximum allowed size" in rejected.text

        accepted = _request(server, size="256x256")
        accepted.raise_for_status()
        payload = accepted.json()
        assert len(payload["data"]) == 1
        image_bytes = base64.b64decode(payload["data"][0]["b64_json"])
        image = Image.open(io.BytesIO(image_bytes))
        assert image.size == (256, 256)
