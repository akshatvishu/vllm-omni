import pytest

from tests.e2e.online_serving.test_qwen3_omni_expansion import (
    get_max_batch_size,
    get_prompt,
    get_system_prompt,
    model,
)
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config


single_gpu_config = modify_stage_config(
    get_deploy_config_path("qwen3_omni_moe.yaml"),
    updates={
        "stages": {
            0: {"devices": "0", "gpu_memory_utilization": 0.5, "max_num_seqs": 5},
            1: {"devices": "0", "gpu_memory_utilization": 0.35, "max_num_seqs": 5},
            2: {"devices": "0", "gpu_memory_utilization": 0.1, "max_num_seqs": 5},
        }
    },
)


@pytest.mark.full_model
@pytest.mark.omni
@pytest.mark.rocm
@pytest.mark.parametrize(
    "omni_server",
    [
        OmniServerParams(
            model=model,
            stage_config_path=single_gpu_config,
            use_stage_cli=True,
            server_args=["--no-async-chunk"],
        )
    ],
    indirect=True,
)
def test_ethan_survives_non_async_stage_handoff(omni_server, openai_client) -> None:
    request_config = {
        "model": omni_server.model,
        "messages": dummy_messages_from_mix_data(
            system_prompt=get_system_prompt(),
            content_text=get_prompt("text"),
        ),
        "stream": True,
        "speaker": "Ethan",
        "key_words": {"text": ["beijing"]},
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())
