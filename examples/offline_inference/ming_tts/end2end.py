"""Offline inference demo for Ming-omni-tts via vLLM Omni."""

import asyncio
import json
import os
import uuid
import wave
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import AsyncOmni, Omni
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    KEY_MAX_DECODE_STEPS,
    SAMPLE_RATE,
    TEXT_EOS_TOKEN_ID,
)
from vllm_omni.model_executor.models.ming_tts.prompt_builder import build_ming_dense_prompt

DEFAULT_MODEL = "inclusionAI/Ming-omni-tts-0.5B"
DEFAULT_STAGE_CONFIG = "vllm_omni/model_executor/stage_configs/ming_tts.yaml"
DEFAULT_STREAM_STAGE_CONFIG = "vllm_omni/model_executor/stage_configs/ming_tts_async_chunk.yaml"
DEFAULT_OUTPUT_DIR = "output_audio"
DEFAULT_SPEECH_PROMPT = "Please generate speech based on the following description.\n"
DEFAULT_MUSIC_PROMPT = "Please generate music based on the following description.\n"

CASE_DEFAULTS = {
    "style": {
        "prompt": DEFAULT_SPEECH_PROMPT,
        "text": "我会一直在这里陪着你，直到你慢慢、慢慢地沉入那个最温柔的梦里……好吗？",
        "instruction": {
            "风格": (
                "这是一种ASMR耳语，属于一种旨在引发特殊感官体验的创意风格。"
                "这个女性使用轻柔的普通话进行耳语，声音气音成分重。"
                "音量极低，紧贴麦克风，语速极慢。"
            )
        },
        "use_zero_spk_emb": True,
        "max_decode_steps": 200,
    },
    "ip": {
        "prompt": DEFAULT_SPEECH_PROMPT,
        "text": "这款产品的名字，叫变态坑爹牛肉丸。",
        "instruction": {"IP": "灵小甄"},
        "use_zero_spk_emb": True,
        "max_decode_steps": 200,
    },
    "bgm": {
        "prompt": DEFAULT_MUSIC_PROMPT,
        "text": "Genre: 电子舞曲. Mood: 自信 / 坚定. Instrument: 架子鼓. Theme: 节日. Duration: 30s.",
        "instruction": None,
        "use_zero_spk_emb": False,
        "max_decode_steps": 400,
    },
    "basic": {
        "prompt": DEFAULT_SPEECH_PROMPT,
        "text": "简单地说，这相当于惠普把消费领域市场拱手相让了。",
        "instruction": {"语速": "快速", "基频": "中", "音量": "高"},
        "requires_ref_audio": True,
        "max_decode_steps": 200,
    },
    "dialect": {
        "prompt": DEFAULT_SPEECH_PROMPT,
        "text": "我觉得社会企业同个人都有责任",
        "instruction": {"方言": "广粤话"},
        "requires_ref_audio": True,
        "max_decode_steps": 200,
    },
    "zero_shot": {
        "prompt": DEFAULT_SPEECH_PROMPT,
        "text": "我们的愿景是构建未来服务业的数字化基础设施，为世界带来更多微小而美好的改变。",
        "instruction": None,
        "requires_ref_audio": True,
        "requires_ref_text": True,
        "max_decode_steps": 200,
    },
}


def _load_reference_waveform(path: str) -> torch.Tensor:
    samples, sample_rate = sf.read(path, dtype="float32")
    waveform = torch.as_tensor(samples, dtype=torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)
    waveform = waveform.reshape(1, -1)
    if int(sample_rate) != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, int(sample_rate), SAMPLE_RATE)
    return waveform


def _load_speaker_embedding(path: str) -> torch.Tensor:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return torch.as_tensor(data, dtype=torch.float32)


def _flatten_audio(audio) -> torch.Tensor:
    if isinstance(audio, list):
        parts = []
        for item in audio:
            tensor = torch.as_tensor(item, dtype=torch.float32).reshape(-1)
            if tensor.numel() > 0:
                parts.append(tensor)
        if not parts:
            return torch.zeros((0,), dtype=torch.float32)
        return torch.cat(parts, dim=0)
    return torch.as_tensor(audio, dtype=torch.float32).reshape(-1)


def _resolve_sr(sr) -> int:
    if isinstance(sr, list):
        sr = sr[-1]
    if hasattr(sr, "item"):
        return int(sr.item())
    return int(sr)


def _write_wav(path: str, audio: torch.Tensor, sample_rate: int) -> None:
    audio = audio.clamp(-1.0, 1.0)
    pcm16 = (audio * 32767.0).round().to(torch.int16).cpu().numpy()
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())


def _extract_audio_output(outputs):
    output = next((item for item in outputs if item.final_output_type == "audio"), None)
    if output is None:
        raise RuntimeError("Expected one final output with final_output_type='audio'")

    multimodal_output = output.multimodal_output or {}
    audio = multimodal_output.get("audio")
    sr = multimodal_output.get("sr")
    if audio is None or sr is None:
        raise RuntimeError("Expected multimodal_output['audio'] and multimodal_output['sr']")

    waveform = _flatten_audio(audio)
    if waveform.numel() == 0:
        raise RuntimeError("Generated audio waveform is empty")
    return waveform, _resolve_sr(sr)


def _build_instruction(args, case):
    if args.instruction_json is not None:
        return json.loads(args.instruction_json)
    if args.instructions is not None:
        return args.instructions
    return case.get("instruction")


def _build_prompt(tokenizer, args):
    case = CASE_DEFAULTS[args.case]
    prompt = args.prompt or case["prompt"]
    text = args.text or case["text"]
    instruction = _build_instruction(args, case)
    prompt_text = args.ref_text
    prompt_waveform = None
    if args.ref_audio:
        prompt_waveform = _load_reference_waveform(args.ref_audio)
    elif case.get("requires_ref_audio"):
        raise RuntimeError(f"--ref-audio is required for case '{args.case}'")

    if case.get("requires_ref_text") and not prompt_text:
        raise RuntimeError(f"--ref-text is required for case '{args.case}'")

    speaker_embedding = _load_speaker_embedding(args.speaker_embedding) if args.speaker_embedding else None
    use_zero_spk_emb = bool(case.get("use_zero_spk_emb", False)) and prompt_waveform is None and speaker_embedding is None

    runtime_controls = {
        KEY_MAX_DECODE_STEPS: args.max_decode_steps or case["max_decode_steps"],
    }
    return build_ming_dense_prompt(
        tokenizer,
        prompt=prompt,
        text=text,
        runtime_controls=runtime_controls,
        instruction=instruction,
        prompt_text=prompt_text,
        prompt_waveform=prompt_waveform,
        speaker_embedding=speaker_embedding,
        use_zero_spk_emb=use_zero_spk_emb,
    )


async def _run_streaming(args, prompt_payload, sampling_params_list, output_path):
    engine = AsyncOmni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
        log_stats=args.log_stats,
    )
    try:
        all_audio_chunks = []
        accumulated_samples = 0
        chunk_idx = 0
        async for stage_output in engine.generate(
            prompt=prompt_payload,
            request_id=str(uuid.uuid4()),
            sampling_params_list=sampling_params_list,
        ):
            multimodal_output = stage_output.multimodal_output or {}
            audio = multimodal_output.get("audio")
            if audio is None:
                continue

            finished = stage_output.finished
            if isinstance(audio, torch.Tensor):
                if finished:
                    audio_chunk = audio[accumulated_samples:].float().detach().cpu()
                else:
                    audio_chunk = audio.float().detach().cpu()
            elif isinstance(audio, list):
                audio_chunk = torch.as_tensor(audio[chunk_idx], dtype=torch.float32).reshape(-1).cpu()
            else:
                audio_chunk = torch.as_tensor(audio, dtype=torch.float32).reshape(-1).cpu()

            accumulated_samples += int(audio_chunk.numel())
            chunk_idx += 1
            if audio_chunk.numel() > 0:
                all_audio_chunks.append(audio_chunk)

        if not all_audio_chunks:
            raise RuntimeError("Streaming Ming example produced no audio chunks")

        waveform = torch.cat(all_audio_chunks, dim=0)
        _write_wav(output_path, waveform, SAMPLE_RATE)
        print(f"Saved streaming output to {output_path}")
    finally:
        engine.shutdown()


def _run_non_streaming(args, prompt_payload, sampling_params_list, output_path):
    engine = Omni(
        model=args.model,
        stage_configs_path=args.stage_configs_path,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
        log_stats=args.log_stats,
    )
    try:
        outputs = engine.generate(
            prompts=[prompt_payload],
            sampling_params_list=sampling_params_list,
            py_generator=False,
        )
        waveform, sample_rate = _extract_audio_output(outputs)
        _write_wav(output_path, waveform, sample_rate)
        print(f"Saved output to {output_path}")
    finally:
        engine.close()


def main():
    parser = FlexibleArgumentParser(description="Offline Ming-omni-tts example")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name or local path")
    parser.add_argument(
        "--stage-configs-path",
        default=None,
        help="Stage config path. Defaults to ming_tts.yaml or ming_tts_async_chunk.yaml when --streaming is set.",
    )
    parser.add_argument("--case", choices=sorted(CASE_DEFAULTS), default="style", help="Built-in demo case")
    parser.add_argument("--text", default=None, help="Override case text")
    parser.add_argument("--prompt", default=None, help="Override the system prompt prefix")
    parser.add_argument("--instructions", default=None, help="Free-form Ming instruction string")
    parser.add_argument(
        "--instruction-json",
        default=None,
        help="Structured Ming instruction JSON, for example '{\"方言\":\"广粤话\"}'",
    )
    parser.add_argument("--ref-audio", default=None, help="Reference audio path for cloning")
    parser.add_argument("--ref-text", default=None, help="Reference transcript for cloning")
    parser.add_argument("--speaker-embedding", default=None, help="Path to a JSON speaker embedding file")
    parser.add_argument("--max-decode-steps", type=int, default=None, help="Override ming_max_decode_steps")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for output wav files")
    parser.add_argument("--output-name", default=None, help="Output wav filename")
    parser.add_argument("--streaming", action="store_true", help="Use AsyncOmni with async_chunk streaming")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code to Omni")
    parser.add_argument("--enforce-eager", action="store_true", help="Pass enforce_eager to Omni")
    parser.add_argument("--log-stats", action="store_true", help="Enable Omni stats logging")
    args = parser.parse_args()

    if args.instructions is not None and args.instruction_json is not None:
        raise RuntimeError("Use either --instructions or --instruction-json, not both")

    if args.stage_configs_path is None:
        args.stage_configs_path = DEFAULT_STREAM_STAGE_CONFIG if args.streaming else DEFAULT_STAGE_CONFIG

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    prompt_payload = _build_prompt(tokenizer, args)

    max_decode_steps = args.max_decode_steps or CASE_DEFAULTS[args.case]["max_decode_steps"]
    sampling_params_list = [
        SamplingParams(
            temperature=0.0,
            max_tokens=max_decode_steps + 1,
            stop_token_ids=[int(TEXT_EOS_TOKEN_ID)],
        ),
        SamplingParams(temperature=0.0, max_tokens=1),
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"ming_{args.case}.wav"
    output_path = str(output_dir / output_name)

    if args.streaming:
        asyncio.run(_run_streaming(args, prompt_payload, sampling_params_list, output_path))
    else:
        _run_non_streaming(args, prompt_payload, sampling_params_list, output_path)


if __name__ == "__main__":
    main()
