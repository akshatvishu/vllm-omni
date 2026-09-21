# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import shlex
from unittest.mock import Mock

import pytest

from tests.examples.offline_inference import test_text_to_image as t2i

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_anima_readme_assets(monkeypatch):
    checkpoint = "/cache with spaces/anima.safetensors"
    components = "/cache with spaces/components"
    download_checkpoint = Mock(return_value=checkpoint)
    download_components = Mock(return_value=components)
    monkeypatch.setattr(t2i, "hf_hub_download", download_checkpoint)
    monkeypatch.setattr(t2i, "snapshot_download", download_components)
    snippet = next(s for s in t2i.README_SNIPPETS if "--model-class-name AnimaPipeline" in s.code)
    runner = Mock()
    runner.run.return_value.assets = []

    t2i.test_text_to_image(snippet, runner)

    runner.run.assert_called_once()
    prepared = runner.run.call_args.args[0]
    argv = shlex.split(prepared.code)

    assert argv[argv.index("--model") + 1] == checkpoint
    assert json.loads(argv[argv.index("--custom-pipeline-args") + 1])["components_path"] == components
    assert "/path/to/" not in prepared.code
    assert prepared.output_file_path == snippet.output_file_path
    assert argv[argv.index("--prompt") + 1] == "A cinematic close-up of a glass teapot on a wooden table."
    download_checkpoint.assert_called_once_with(
        repo_id="circlestone-labs/Anima",
        filename="split_files/diffusion_models/anima-base-v1.0.safetensors",
    )
    download_components.assert_called_once_with(
        repo_id="circlestone-labs/Anima-Base-v1.0-Diffusers",
        allow_patterns=["text_encoder/*", "vae/*", "tokenizer/*", "t5_tokenizer/*", "scheduler/*"],
    )


def test_other_readme_examples_do_not_download_anima(monkeypatch):
    download_checkpoint = Mock()
    download_components = Mock()
    monkeypatch.setattr(t2i, "hf_hub_download", download_checkpoint)
    monkeypatch.setattr(t2i, "snapshot_download", download_components)

    for snippet in t2i.README_SNIPPETS:
        if "--model-class-name AnimaPipeline" not in snippet.code:
            assert t2i._prepare_anima_snippet(snippet) is snippet

    download_checkpoint.assert_not_called()
    download_components.assert_not_called()
