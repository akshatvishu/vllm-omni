# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
from pathlib import Path

PIPELINE_PATH = Path("vllm_omni/diffusion/models/omnivoice/pipeline_omnivoice.py")

REPLACEMENTS = (
    (
        """        decoded_chunks: list[torch.Tensor] = []
        audio_copy_stream: torch.Stream | None = None
        fixed_ref_text = ref_text
""",
        """        decoded_chunks: list[torch.Tensor] = []
        fixed_ref_text = ref_text
""",
    ),
    (
        """            decoded_audio = self.decoder(tokens)
            if decoded_audio.device.type != "cpu" and self.pin_memory and audio_copy_stream is None:
                audio_copy_stream = torch.Stream(device=decoded_audio.device)
            decoded_chunks.append(_copy_audio_to_cpu(decoded_audio, audio_copy_stream))
""",
        """            decoded_audio = self.decoder(tokens)
            decoded_chunks.append(decoded_audio)
""",
    ),
    (
        """        if audio_copy_stream is not None:
            audio_copy_stream.synchronize()
        return join_audio_chunks(decoded_chunks, self.sample_rate)
""",
        """        decoded_chunks = [chunk.detach().cpu() for chunk in decoded_chunks]
        return join_audio_chunks(decoded_chunks, self.sample_rate)
""",
    ),
)


def prepare_gpu_retention_baseline(repo_root: Path) -> None:
    pipeline_path = repo_root / PIPELINE_PATH
    source = pipeline_path.read_text(encoding="utf-8")
    baseline = source
    for current_code, baseline_code in REPLACEMENTS:
        if baseline.count(current_code) != 1:
            raise ValueError(f"expected code was not found exactly once in {pipeline_path}")
        baseline = baseline.replace(current_code, baseline_code)
    pipeline_path.write_text(baseline, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore OmniVoice GPU chunk retention for an A/B baseline")
    parser.add_argument("--repo-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_gpu_retention_baseline(args.repo_root)


if __name__ == "__main__":
    main()
