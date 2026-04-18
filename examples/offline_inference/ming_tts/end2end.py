# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference demo for Ming-omni-tts via vLLM Omni."""

import asyncio
import json
import os
import time
import uuid
import wave
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni import AsyncOmni, Omni
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    KEY_CFG,
    KEY_MAX_DECODE_STEPS,
    KEY_SIGMA,
    KEY_SPEAKER_EMBEDDING,
    KEY_TEMPERATURE,
    SAMPLE_RATE,
    TEXT_EOS_TOKEN_ID,
)
from vllm_omni.model_executor.models.ming_tts.prompt_builder import build_ming_dense_prompt
from vllm_omni.model_executor.models.ming_tts.speaker_extractor import MingSpeakerEmbeddingExtractor

DEFAULT_MODEL = "inclusionAI/Ming-omni-tts-0.5B"
DEFAULT_STAGE_CONFIG = "vllm_omni/model_executor/stage_configs/ming_tts.yaml"
DEFAULT_STREAM_STAGE_CONFIG = "vllm_omni/model_executor/stage_configs/ming_tts_async_chunk.yaml"
DEFAULT_OUTPUT_DIR = "output_audio"
DEFAULT_SPEECH_PROMPT = "Please generate speech based on the following description.\n"
DEFAULT_MUSIC_PROMPT = "Please generate music based on the following description.\n"
DEFAULT_PODCAST_TEXT = (
    " speaker_1:你可以说一下，就大概说一下，可能虽然我也不知道，我看过那部电影没有。\n"
    " speaker_2:就是那个叫什么，变相一节课的嘛。\n"
    " speaker_1:嗯。\n"
    " speaker_2:一部搞笑的电影。\n"
    " speaker_1:一部搞笑的。\n"
)
DEFAULT_PODCAST_PROMPT_TEXT = (
    " speaker_1:并且我们还要进行每个月还要考核 笔试的话还要进行笔试，做个，当服务员还要去笔试了\n"
    " speaker_2:对啊，这真的很奇怪，就是 单纯的因，单纯自己工资不高，只是因为可能人家那个店比较出名一点，就对你苛刻要求\n"
)

CASE_DEFAULTS = {
    "style": {
        "prompt": DEFAULT_SPEECH_PROMPT,
        "text": "我会一直在这里陪着你，直到你慢慢、慢慢地沉入那个最温柔的梦里……好吗？",
        "instruction": {
            "风格": (
                "这是一种ASMR耳语，属于一种旨在引发特殊感官体验的创意风格。"
                "这个女性使用轻柔的普通话进行耳语，声音气音成分重。"
                "音量极低，紧贴麦克风，语速极慢，旨在制造触发听者颅内快感的声学刺激。"
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
    "tta": {
        "prompt": "Please generate audio events based on given text.\n",
        "text": "Thunder and a gentle rain",
        "instruction": None,
        "use_zero_spk_emb": False,
        "max_decode_steps": 200,
        "cfg": 4.5,
        "sigma": 0.3,
        "temperature": 2.5,
    },
    "emotion": {
        "prompt": DEFAULT_SPEECH_PROMPT,
        "text": "我竟然抢到了陈奕迅的演唱会门票！太棒了！终于可以现场听一听他的歌声了！",
        "instruction": {"情感": "高兴"},
        "requires_ref_audio": True,
        "auto_extract_speaker_embeddings": True,
        "max_decode_steps": 200,
    },
    "basic": {
        "prompt": DEFAULT_SPEECH_PROMPT,
        "text": "简单地说，这相当于惠普把消费领域市场拱手相让了。",
        "instruction": {"语速": "快速", "基频": "中", "音量": "高"},
        "requires_ref_audio": True,
        "auto_extract_speaker_embeddings": True,
        "max_decode_steps": 200,
    },
    "dialect": {
        "prompt": DEFAULT_SPEECH_PROMPT,
        "text": "我觉得社会企业同个人都有责任",
        "instruction": {"方言": "广粤话"},
        "requires_ref_audio": True,
        "auto_extract_speaker_embeddings": True,
        "max_decode_steps": 200,
    },
    "zero_shot": {
        "prompt": DEFAULT_SPEECH_PROMPT,
        "text": "我们的愿景是构建未来服务业的数字化基础设施，为世界带来更多微小而美好的改变。",
        "instruction": None,
        "requires_ref_audio": True,
        "requires_ref_text": True,
        "auto_extract_speaker_embeddings": True,
        "max_decode_steps": 200,
    },
    "podcast": {
        "prompt": DEFAULT_SPEECH_PROMPT,
        "text": DEFAULT_PODCAST_TEXT,
        "instruction": None,
        "prompt_text": DEFAULT_PODCAST_PROMPT_TEXT,
        "requires_ref_audio_count": 2,
        "auto_extract_speaker_embeddings": True,
        "max_decode_steps": 200,
    },
    "speech_bgm": {
        "prompt": DEFAULT_SPEECH_PROMPT,
        "text": "此次业绩下滑原因，可归结为企业停止服务某些品牌，而带来的负面影响。",
        "instruction": {
            "BGM": {
                "Genre": "当代古典音乐.",
                "Mood": "温暖 / 友善.",
                "Instrument": "电吉他",
                "Theme": "节日.",
                "SNR": 10.0,
                "ENV": None,
            }
        },
        "requires_ref_audio": True,
        "auto_extract_speaker_embeddings": True,
        "max_decode_steps": 200,
    },
    "speech_sound": {
        "prompt": DEFAULT_SPEECH_PROMPT,
        "text": "此次业绩下滑原因，可归结为企业停止服务某些品牌，而带来的负面影响。",
        "instruction": {
            "BGM": {
                "ENV": "Birds chirping",
                "SNR": 10.0,
                "Genre": None,
                "Mood": None,
                "Instrument": None,
                "Theme": None,
            }
        },
        "requires_ref_audio": True,
        "auto_extract_speaker_embeddings": True,
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


def _resolve_reference_inputs(args, case):
    if args.ref_audio is not None and args.ref_audio_paths is not None:
        raise RuntimeError("Use either --ref-audio or --ref-audio-paths, not both")

    if args.ref_audio_paths is not None:
        ref_audio_paths = list(args.ref_audio_paths)
    elif args.ref_audio is not None:
        ref_audio_paths = [args.ref_audio]
    else:
        ref_audio_paths = []

    required_count = int(case.get("requires_ref_audio_count", 0))
    if required_count > 0:
        if len(ref_audio_paths) < required_count:
            raise RuntimeError(
                f"Case '{args.case}' requires at least {required_count} reference audio paths via --ref-audio-paths"
            )
    elif case.get("requires_ref_audio") and not ref_audio_paths:
        raise RuntimeError(f"--ref-audio is required for case '{args.case}'")

    if not ref_audio_paths:
        return None
    if len(ref_audio_paths) == 1:
        return _load_reference_waveform(ref_audio_paths[0])
    return [_load_reference_waveform(path) for path in ref_audio_paths]


def _resolve_reference_audio_paths(args):
    if args.ref_audio is not None and args.ref_audio_paths is not None:
        raise RuntimeError("Use either --ref-audio or --ref-audio-paths, not both")
    if args.ref_audio_paths is not None:
        return list(args.ref_audio_paths)
    if args.ref_audio is not None:
        return [args.ref_audio]
    return []


def _resolve_speaker_embedding(args, case, ref_audio_paths):
    if args.speaker_embedding:
        return _load_speaker_embedding(args.speaker_embedding)

    should_extract = bool(case.get("auto_extract_speaker_embeddings", False) or args.extract_speaker_embeddings)
    if not should_extract or not ref_audio_paths:
        return None

    extractor = MingSpeakerEmbeddingExtractor(args.model)
    embeddings = extractor.extract_many(ref_audio_paths)
    if not embeddings:
        raise RuntimeError("Speaker extraction produced no embeddings")
    if len(embeddings) == 1:
        return embeddings[0]
    return torch.stack(embeddings, dim=0)


def _coerce_audio_tensor(audio, *, async_chunk: bool) -> torch.Tensor:
    if isinstance(audio, list):
        if async_chunk:
            parts = []
            for item in audio:
                tensor = torch.as_tensor(item, dtype=torch.float32).reshape(-1)
                if tensor.numel() > 0:
                    parts.append(tensor)
            if not parts:
                return torch.zeros((0,), dtype=torch.float32)
            return torch.cat(parts, dim=0)

        for item in reversed(audio):
            tensor = torch.as_tensor(item, dtype=torch.float32).reshape(-1)
            if tensor.numel() > 0:
                return tensor
        return torch.zeros((0,), dtype=torch.float32)

    return torch.as_tensor(audio, dtype=torch.float32).reshape(-1)


def _resolve_sr(sr) -> int:
    if isinstance(sr, list):
        sr = sr[-1]
    if hasattr(sr, "item"):
        return int(sr.item())
    return int(sr)


def _extract_sample_rate(multimodal_output: dict) -> int:
    sr = multimodal_output.get("sr")
    if sr is None:
        raise RuntimeError("Expected multimodal_output['sr']")
    return _resolve_sr(sr)


def _write_wav(path: str, audio: torch.Tensor, sample_rate: int) -> None:
    audio = audio.clamp(-1.0, 1.0)
    pcm16 = (audio * 32767.0).round().to(torch.int16).cpu().numpy()
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())


def _request_index(request_id: str | None, fallback: int) -> int:
    try:
        return int(request_id)
    except (TypeError, ValueError):
        if isinstance(request_id, str):
            head = request_id.split("_", 1)[0]
            if head.isdigit():
                return int(head)
    return fallback


def _audio_summary(audio: torch.Tensor, sample_rate: int) -> dict:
    waveform = audio.detach().cpu().reshape(-1).to(torch.float32)
    return {
        "sample_rate": int(sample_rate),
        "num_samples": int(waveform.numel()),
        "duration_seconds": float(waveform.numel()) / float(sample_rate),
        "max_abs_amplitude": float(waveform.abs().max().item()) if waveform.numel() > 0 else 0.0,
    }


def _resolve_output_name(output_name: str | None, case: str, index: int, total: int) -> str:
    if total == 1:
        return output_name or f"ming_{case}.wav"
    base = Path(output_name or f"ming_{case}.wav")
    return f"{base.stem}_{index:05d}{base.suffix or '.wav'}"


def _resolve_stats_log_file(args) -> str | None:
    if not args.log_stats:
        return None
    if args.stats_log_file:
        return args.stats_log_file
    base = Path(args.output_name or f"ming_{args.case}.wav").stem
    return str(Path(args.output_dir) / f"{base}_pipeline.log")


def _resolve_metadata_json(args) -> str | None:
    if args.metadata_json:
        return args.metadata_json
    if args.log_stats:
        base = Path(args.output_name or f"ming_{args.case}.wav").stem
        return str(Path(args.output_dir) / f"{base}_manifest.json")
    return None


def _build_manifest(args, prompt_payload, stats_log_file: str | None, outputs: list[dict]) -> dict:
    additional_information = {}
    if isinstance(prompt_payload, dict):
        additional_information = dict(prompt_payload.get("additional_information", {}))
    return {
        "model": args.model,
        "case": args.case,
        "streaming": bool(args.streaming),
        "stage_configs_path": args.stage_configs_path,
        "enforce_eager": bool(args.enforce_eager),
        "num_prompts": int(args.num_prompts),
        "log_stats": bool(args.log_stats),
        "stats_log_file": stats_log_file,
        "prompt_text": additional_information.get("prompt_text"),
        "instruction": additional_information.get("instruction"),
        "speaker_embedding_shape": (
            list(additional_information[KEY_SPEAKER_EMBEDDING].shape)
            if KEY_SPEAKER_EMBEDDING in additional_information
            and hasattr(additional_information[KEY_SPEAKER_EMBEDDING], "shape")
            else None
        ),
        "outputs": outputs,
        "generated_at_unix": time.time(),
    }


def _build_engine_kwargs(args, stats_log_file: str | None) -> dict:
    kwargs = {
        "model": args.model,
        "stage_configs_path": args.stage_configs_path,
        "enforce_eager": args.enforce_eager,
        "trust_remote_code": args.trust_remote_code,
        "log_stats": args.log_stats,
        "stage_init_timeout": args.stage_init_timeout,
        "init_timeout": args.init_timeout,
        "batch_timeout": args.batch_timeout,
        "shm_threshold_bytes": args.shm_threshold_bytes,
        "worker_backend": args.worker_backend,
    }
    if stats_log_file is not None:
        kwargs["log_file"] = stats_log_file
    if args.ray_address is not None:
        kwargs["ray_address"] = args.ray_address
    return kwargs


def _extract_audio_output(outputs, *, async_chunk: bool):
    output = next((item for item in outputs if item.final_output_type == "audio"), None)
    if output is None:
        raise RuntimeError("Expected one final output with final_output_type='audio'")

    multimodal_output = output.multimodal_output or {}
    audio = multimodal_output.get("audio")
    sr = multimodal_output.get("sr")
    if audio is None or sr is None:
        raise RuntimeError("Expected multimodal_output['audio'] and multimodal_output['sr']")

    waveform = _coerce_audio_tensor(audio, async_chunk=async_chunk)
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
    prompt_text = args.ref_text if args.ref_text is not None else case.get("prompt_text")
    ref_audio_paths = _resolve_reference_audio_paths(args)
    prompt_waveform = _resolve_reference_inputs(args, case) if prompt_text is not None else None

    required_count = int(case.get("requires_ref_audio_count", 0))
    if required_count > 0 and len(ref_audio_paths) < required_count:
        raise RuntimeError(
            f"Case '{args.case}' requires at least {required_count} reference audio paths via --ref-audio-paths"
        )
    if required_count <= 0 and case.get("requires_ref_audio") and not ref_audio_paths:
        raise RuntimeError(f"--ref-audio is required for case '{args.case}'")

    if case.get("requires_ref_text") and not prompt_text:
        raise RuntimeError(f"--ref-text is required for case '{args.case}'")

    speaker_embedding = _resolve_speaker_embedding(args, case, ref_audio_paths)
    use_zero_spk_emb = (
        bool(case.get("use_zero_spk_emb", False)) and prompt_waveform is None and speaker_embedding is None
    )

    runtime_controls = {
        KEY_MAX_DECODE_STEPS: args.max_decode_steps or case["max_decode_steps"],
    }
    if "cfg" in case:
        runtime_controls[KEY_CFG] = case["cfg"]
    if "sigma" in case:
        runtime_controls[KEY_SIGMA] = case["sigma"]
    if "temperature" in case:
        runtime_controls[KEY_TEMPERATURE] = case["temperature"]
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


async def _run_streaming(args, prompt_payload, sampling_params_list, output_dir, stats_log_file):
    engine = AsyncOmni(**_build_engine_kwargs(args, stats_log_file))
    try:
        all_audio_chunks = []
        accumulated_samples = 0
        chunk_idx = 0
        start_time = time.time()
        chunk_times = []
        ttfp_seconds = None
        final_stage_output = None
        async for stage_output in engine.generate(
            prompt=prompt_payload,
            request_id=str(uuid.uuid4()),
            sampling_params_list=sampling_params_list,
        ):
            final_stage_output = stage_output
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
                now = time.time()
                if ttfp_seconds is None:
                    ttfp_seconds = now - start_time
                chunk_times.append(now)
                all_audio_chunks.append(audio_chunk)

        if not all_audio_chunks:
            raise RuntimeError("Streaming Ming example produced no audio chunks")

        waveform = torch.cat(all_audio_chunks, dim=0)
        output_name = _resolve_output_name(args.output_name, args.case, 0, 1)
        output_path = str(Path(output_dir) / output_name)
        _write_wav(output_path, waveform, SAMPLE_RATE)
        summary = {
            "request_id": getattr(final_stage_output, "request_id", None),
            "stage_id": getattr(final_stage_output, "stage_id", None),
            "output_path": output_path,
            "stage_durations": getattr(final_stage_output, "stage_durations", {}),
            "peak_memory_mb": getattr(final_stage_output, "peak_memory_mb", 0.0),
            "ttfp_seconds": ttfp_seconds,
            "mean_inter_chunk_seconds": (
                sum(t1 - t0 for t0, t1 in zip(chunk_times, chunk_times[1:])) / (len(chunk_times) - 1)
                if len(chunk_times) > 1
                else None
            ),
        }
        summary.update(_audio_summary(waveform, SAMPLE_RATE))
        print(f"Saved streaming output to {output_path}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return [summary]
    finally:
        engine.shutdown()


def _run_non_streaming(args, prompt_payload, sampling_params_list, output_dir, stats_log_file):
    engine = Omni(**_build_engine_kwargs(args, stats_log_file))
    try:
        outputs = engine.generate(
            prompts=[prompt_payload for _ in range(args.num_prompts)],
            sampling_params_list=sampling_params_list,
            py_generator=False,
        )
        summaries = []
        for fallback_index, output in enumerate(outputs):
            if output.final_output_type != "audio":
                continue
            multimodal_output = output.multimodal_output or {}
            waveform = _coerce_audio_tensor(multimodal_output.get("audio"), async_chunk=False)
            sample_rate = _extract_sample_rate(multimodal_output)
            request_index = _request_index(output.request_id, fallback_index)
            output_name = _resolve_output_name(args.output_name, args.case, request_index, args.num_prompts)
            output_path = str(Path(output_dir) / output_name)
            _write_wav(output_path, waveform, sample_rate)
            summary = {
                "request_id": output.request_id,
                "stage_id": output.stage_id,
                "output_path": output_path,
                "stage_durations": output.stage_durations,
                "peak_memory_mb": output.peak_memory_mb,
            }
            summary.update(_audio_summary(waveform, sample_rate))
            summaries.append(summary)
            print(f"Saved output to {output_path}")
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not summaries:
            raise RuntimeError("Non-streaming Ming example produced no audio outputs")
        return summaries
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
        help='Structured Ming instruction JSON, for example \'{"方言":"广粤话"}\'',
    )
    parser.add_argument("--ref-audio", default=None, help="Reference audio path for cloning")
    parser.add_argument(
        "--ref-audio-paths",
        nargs="+",
        default=None,
        help="Multiple reference audio paths, used by multi-speaker cases like podcast",
    )
    parser.add_argument("--ref-text", default=None, help="Reference transcript for cloning")
    parser.add_argument("--speaker-embedding", default=None, help="Path to a JSON speaker embedding file")
    parser.add_argument(
        "--extract-speaker-embeddings",
        action="store_true",
        help="Extract 192-d Ming speaker embeddings from --ref-audio or --ref-audio-paths using campplus.onnx",
    )
    parser.add_argument("--max-decode-steps", type=int, default=None, help="Override ming_max_decode_steps")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for output wav files")
    parser.add_argument("--output-name", default=None, help="Output wav filename")
    parser.add_argument("--num-prompts", type=int, default=1, help="Repeat the same prompt N times")
    parser.add_argument("--streaming", action="store_true", help="Use AsyncOmni with async_chunk streaming")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code to Omni")
    parser.add_argument("--enforce-eager", action="store_true", help="Pass enforce_eager to Omni")
    parser.add_argument(
        "--log-stats", "--enable-stats", dest="log_stats", action="store_true", help="Enable Omni stats logging"
    )
    parser.add_argument("--stats-log-file", default=None, help="Optional path for the Omni stats log file")
    parser.add_argument("--metadata-json", default=None, help="Optional path for a run manifest JSON file")
    parser.add_argument(
        "--stage-init-timeout", type=int, default=300, help="Per-stage initialization timeout in seconds"
    )
    parser.add_argument("--init-timeout", type=int, default=600, help="Total initialization timeout in seconds")
    parser.add_argument("--batch-timeout", type=int, default=5, help="Batch timeout in seconds")
    parser.add_argument("--shm-threshold-bytes", type=int, default=65536, help="Shared memory threshold in bytes")
    parser.add_argument(
        "--worker-backend",
        type=str,
        default="multi_process",
        choices=["multi_process", "ray"],
        help="Worker backend",
    )
    parser.add_argument("--ray-address", default=None, help="Ray cluster address when --worker-backend ray is used")
    args = parser.parse_args()

    if args.instructions is not None and args.instruction_json is not None:
        raise RuntimeError("Use either --instructions or --instruction-json, not both")
    if args.num_prompts < 1:
        raise RuntimeError("--num-prompts must be at least 1")
    if args.streaming and args.num_prompts != 1:
        raise RuntimeError("--streaming currently supports exactly one prompt")

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
    stats_log_file = _resolve_stats_log_file(args)

    if args.streaming:
        summaries = asyncio.run(_run_streaming(args, prompt_payload, sampling_params_list, output_dir, stats_log_file))
    else:
        summaries = _run_non_streaming(args, prompt_payload, sampling_params_list, output_dir, stats_log_file)

    metadata_json = _resolve_metadata_json(args)
    manifest = _build_manifest(args, prompt_payload, stats_log_file, summaries)
    if metadata_json is not None:
        Path(metadata_json).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved run manifest to {metadata_json}")


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
