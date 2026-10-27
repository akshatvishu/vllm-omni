# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""One recorded HTTP request; PNG decoding is outside the HTTP timer."""

import contextlib
import time

import pybase64 as base64
from provenance import digest, write


def request(case, assets, source, output, request_id, phase, timeout=1800):
    import requests
    from PIL import Image

    output.mkdir(parents=True, exist_ok=False)
    payload = dict(case["request"])
    payload.setdefault("generator_device", "cuda")
    payload.update(model="Qwen/Qwen-Image-2.1", response_format="b64_json", output_format="png")
    if "width" in payload:
        payload["size"] = f"{payload.pop('width')}x{payload.pop('height')}"
    payload = {k: v for k, v in payload.items() if v is not None}
    refs = [assets[x] for x in case.get("reference_assets", [])]
    write(
        output / "request.json",
        {"id": request_id, "case": case["id"], "phase": phase, "payload": payload, "references": refs},
    )
    result = {"id": request_id, "case": case["id"], "phase": phase, "success": False}
    try:
        with contextlib.ExitStack() as stack:
            files = []
            for ref in refs:
                path = source / ref["path"]
                if digest(path) != ref["sha256"]:
                    raise RuntimeError(f"asset hash mismatch: {path}")
                files.append(("image", (path.name, stack.enter_context(path.open("rb")), "image/png")))
            start = time.perf_counter()
            result["start_time_ns"] = time.time_ns()
            kwargs = {"data": payload, "files": files} if files else {"json": payload}
            response = requests.post(
                "http://127.0.0.1:8091/v1/images/" + ("edits" if files else "generations"),
                headers={"X-Request-Id": request_id},
                timeout=timeout,
                **kwargs,
            )
            result.update(
                http_seconds=time.perf_counter() - start, end_time_ns=time.time_ns(), status=response.status_code
            )
        (output / "response.json").write_text(response.text)
        response.raise_for_status()
        body = response.json()
        if len(body.get("data", [])) != 1:
            raise RuntimeError("expected exactly one image")
        raw = base64.b64decode(body["data"][0]["b64_json"], validate=True)
        image = output / "image.png"
        image.write_bytes(raw)
        with Image.open(image) as decoded:
            decoded.load()
            if decoded.format != "PNG":
                raise RuntimeError("output is not PNG")
            result.update(image_mode=decoded.mode, image_size=list(decoded.size), png_sha256=digest(image))
        result["success"] = True
    except Exception as exc:
        result["error"] = repr(exc)
    write(output / "result.json", result)
    return result
