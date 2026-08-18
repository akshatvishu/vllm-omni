#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import traceback
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe issue 6301 chat-completion response compatibility.")
    parser.add_argument("--suite", choices=("qwen", "zimage"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


class Probe:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
        self.client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
        self.request_dir = args.output_dir / "requests"
        self.response_dir = args.output_dir / "responses"
        self.summary_path = args.output_dir / "summary.json"
        self.results: list[dict[str, Any]] = []

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as output:
            for value in values:
                output.write(json.dumps(value, ensure_ascii=False) + "\n")

    @staticmethod
    def _modalities(document: dict[str, Any], *, streaming: bool) -> tuple[set[str], int]:
        modalities: set[str] = set()
        image_count = 0

        documents = [document]
        if streaming:
            documents = document.get("events", [])

        def inspect_content(content: Any) -> None:
            nonlocal image_count
            if isinstance(content, str):
                stripped = content.strip()
                if stripped.startswith(("[", "{")):
                    try:
                        inspect_content(json.loads(stripped))
                        return
                    except json.JSONDecodeError:
                        pass
                if content:
                    modalities.add("text")
                return
            if isinstance(content, list):
                for part in content:
                    inspect_content(part)
                return
            if not isinstance(content, dict):
                return

            part_type = content.get("type")
            if part_type in {"text", "output_text"} or "text" in content:
                modalities.add("text")
            if part_type in {"audio", "output_audio"} or "audio" in content:
                modalities.add("audio")
            if part_type in {"image", "image_url", "output_image"} or "image" in content:
                modalities.add("image")
                image_count += 1
            for key in ("text", "audio", "image", "video", "content"):
                value = content.get(key)
                if isinstance(value, (list, dict)):
                    inspect_content(value)

        for item in documents:
            declared_modality = item.get("modality")
            if declared_modality:
                modalities.add(str(declared_modality))
            for choice in item.get("choices", []):
                message = choice.get("delta" if streaming else "message", {}) or {}
                if message.get("audio") is not None:
                    modalities.add("audio")
                inspect_content(message.get("content"))

        return modalities, image_count

    def _raw(self, name: str, payload: dict[str, Any], *, streaming: bool) -> dict[str, Any]:
        request_path = self.request_dir / f"{name}.json"
        response_path = self.response_dir / f"{name}.{'jsonl' if streaming else 'json'}"
        self._write_json(request_path, payload)
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.args.timeout) as response:
                if streaming:
                    events: list[dict[str, Any]] = []
                    done = False
                    for encoded_line in response:
                        line = encoded_line.decode("utf-8").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line.removeprefix("data:").strip()
                        if data == "[DONE]":
                            done = True
                            break
                        events.append(json.loads(data))
                    self._write_jsonl(response_path, events)
                    return {"events": events, "done": done, "http_status": response.status}

                document = json.loads(response.read())
                self._write_json(response_path, document)
                return {**document, "_http_status": response.status}
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            response_path.write_text(body, encoding="utf-8")
            raise RuntimeError(f"HTTP {error.code}: {body}") from error

    def _typed(self, name: str, payload: dict[str, Any], *, streaming: bool) -> dict[str, Any]:
        request_path = self.request_dir / f"{name}.json"
        response_path = self.response_dir / f"{name}.{'jsonl' if streaming else 'json'}"
        self._write_json(request_path, payload)

        known = {key: payload[key] for key in ("model", "messages", "stream") if key in payload}
        extra_body = {key: value for key, value in payload.items() if key not in known}
        response = self.client.chat.completions.create(**known, extra_body=extra_body)
        if streaming:
            chunks = list(response)
            events = [chunk.model_dump(mode="json", exclude_none=False, warnings=False) for chunk in chunks]
            self._write_jsonl(response_path, events)
            for chunk in chunks:
                chunk.model_dump(mode="json", exclude_none=False, warnings="error")
            return {"events": events, "done": True}

        document = response.model_dump(mode="json", exclude_none=False, warnings=False)
        self._write_json(response_path, document)
        response.model_dump(mode="json", exclude_none=False, warnings="error")
        return document

    def _record(
        self,
        name: str,
        operation: Callable[[], dict[str, Any]],
        expected: set[str],
        *,
        streaming: bool,
        images: int = 0,
    ) -> bool:
        try:
            document = operation()
            modalities, image_count = self._modalities(document, streaming=streaming)
            missing = sorted(expected - modalities)
            errors = []
            if missing:
                errors.append(f"missing modalities: {missing}")
            if images and image_count != images:
                errors.append(f"expected {images} images, found {image_count}")
            if streaming and not document.get("done"):
                errors.append("stream ended without [DONE]")
            passed = not errors
            result = {
                "name": name,
                "passed": passed,
                "modalities": sorted(modalities),
                "image_count": image_count,
                "error": "; ".join(errors) if errors else None,
            }
        except Exception as error:  # Keep the remaining matrix cells running.
            passed = False
            result = {
                "name": name,
                "passed": False,
                "modalities": [],
                "image_count": 0,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }

        self.results.append(result)
        state = "PASS" if passed else "FAIL"
        print(f"{state:4} {name}: {result.get('error') or 'ok'}", flush=True)
        return passed

    def _qwen_payload(self, *, streaming: bool) -> dict[str, Any]:
        return {
            "model": self.args.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Qwen, a virtual human capable of generating text and speech. "
                        "Answer briefly and speak the answer."
                    ),
                },
                {"role": "user", "content": "What is the capital of China? Answer in one short sentence."},
            ],
            "modalities": ["text", "audio"],
            "stream": streaming,
            "max_tokens": 64,
            "temperature": 0.0,
        }

    def _zimage_payload(self) -> dict[str, Any]:
        return {
            "model": self.args.model,
            "messages": [{"role": "user", "content": "An orange cat sitting on a blue chair."}],
            "modalities": ["image"],
            "stream": False,
            "height": 256,
            "width": 256,
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "num_outputs_per_prompt": 2,
            "seed": 42,
        }

    def _run_batch(self, *, client_kind: str, streaming: bool) -> None:
        payload = self._qwen_payload(streaming=streaming)
        operation = self._raw if client_kind == "raw" else self._typed
        stream_name = "stream" if streaming else "nonstream"

        def run_one(index: int) -> bool:
            name = f"qwen_{client_kind}_{stream_name}_batch_{index}"
            return self._record(
                name,
                lambda: operation(name, payload, streaming=streaming),
                {"text", "audio"},
                streaming=streaming,
            )

        with ThreadPoolExecutor(max_workers=self.args.batch_size) as executor:
            futures = [executor.submit(run_one, index) for index in range(self.args.batch_size)]
            for future in as_completed(futures):
                future.result()

    def run_qwen(self) -> None:
        for client_kind, operation in (("raw", self._raw), ("typed", self._typed)):
            for streaming in (False, True):
                stream_name = "stream" if streaming else "nonstream"
                name = f"qwen_{client_kind}_{stream_name}_single"
                payload = self._qwen_payload(streaming=streaming)
                self._record(
                    name,
                    lambda operation=operation, name=name, payload=payload, streaming=streaming: operation(
                        name, payload, streaming=streaming
                    ),
                    {"text", "audio"},
                    streaming=streaming,
                )
                self._run_batch(client_kind=client_kind, streaming=streaming)

    def run_zimage(self) -> None:
        payload = self._zimage_payload()
        for client_kind, operation in (("raw", self._raw), ("typed", self._typed)):
            name = f"zimage_{client_kind}_nonstream_two_images"
            self._record(
                name,
                lambda operation=operation, name=name: operation(name, payload, streaming=False),
                {"image"},
                streaming=False,
                images=2,
            )

    def finish(self) -> int:
        prior: list[dict[str, Any]] = []
        if self.summary_path.exists():
            prior = json.loads(self.summary_path.read_text(encoding="utf-8")).get("results", [])
        combined = [*prior, *self.results]
        passed = sum(bool(result["passed"]) for result in combined)
        failed = len(combined) - passed
        self._write_json(
            self.summary_path,
            {"passed": passed, "failed": failed, "results": combined},
        )
        print(f"SUMMARY passed={passed} failed={failed} path={self.summary_path}", flush=True)
        return int(any(not result["passed"] for result in self.results))


def main() -> int:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 to test concurrency")
    probe = Probe(args)
    if args.suite == "qwen":
        probe.run_qwen()
    else:
        probe.run_zimage()
    return probe.finish()


if __name__ == "__main__":
    raise SystemExit(main())
