# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""One fresh server process/round, immutable output directory, saved B0-compatible inputs."""

import argparse
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from client import request
from provenance import collect, digest, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "checkpoint", "inputs-dir", "expected-sha", "run-dir", "cache-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=["eager", "compiled"], default="compiled")
    parser.add_argument("--cases", nargs="+", default=["t2i-accuracy"])
    parser.add_argument("--historical", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--stage-bench", action="store_true")
    parser.add_argument("--replay-dir")
    parser.add_argument("--variant", default="B0")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.stage_bench and not args.diagnostic:
        parser.error("--stage-bench requires --diagnostic")
    if args.diagnostic and args.historical:
        parser.error("historical captures require scheduling-specific instrumentation")
    if args.samples < 1 or args.warmups < 0:
        parser.error("invalid sample count")
    source = Path(args.source_dir).resolve()
    inputs = Path(args.inputs_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    run = Path(args.run_dir).resolve()
    harness = Path(__file__).resolve().parent
    cache = Path(args.cache_dir).resolve()
    run.mkdir(parents=True, exist_ok=False)
    server = None
    stop = threading.Event()
    monitor = None
    events = []

    def mark(phase):
        events.append({"phase": phase, "time_ns": time.time_ns()})
        write(run / "phases.json", events)

    def telemetry():
        with (run / "telemetry.jsonl").open("w") as stream:
            while not stop.is_set():
                sample = {"time_ns": time.time_ns()}
                for label, command in [
                    ("processes", ["ps", "-eo", "pid,pgid,rss,args"]),
                    (
                        "device",
                        ["rocm-smi", "--showmeminfo", "vram", "--showpower", "--showtemp", "--showclocks", "--json"],
                    ),
                ]:
                    try:
                        result = subprocess.run(command, capture_output=True, text=True, timeout=8)
                        value = result.stdout
                        if label == "processes":
                            value = "\n".join(
                                line
                                for line in value.splitlines()
                                if len(line.split()) > 2 and line.split()[1] == str(server.pid)
                            )
                        sample[label] = {"returncode": result.returncode, "stdout": value, "stderr": result.stderr}
                    except Exception as exc:
                        sample[label] = {"error": repr(exc)}
                stream.write(json.dumps(sample) + "\n")
                stream.flush()
                stop.wait(2)

    try:
        mark("preflight")
        if not checkpoint.is_dir():
            raise RuntimeError("checkpoint directory missing")
        if not (harness / "hook/sitecustomize.py").is_file():
            raise RuntimeError("required prompt-cache/runtime hook missing")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", 8091)) == 0:
                raise RuntimeError("port 8091 already in use; refusing unknown server")
        provenance = collect(source, args.expected_sha, inputs, harness)
        provenance["arguments"] = vars(args)
        provenance["checkpoint"] = str(checkpoint)
        provenance["checkpoint_files"] = {
            str(p.relative_to(checkpoint)): {"bytes": p.stat().st_size, "sha256": digest(p)}
            for p in checkpoint.rglob("*")
            if p.is_file() and p.suffix in (".json", ".safetensors")
        }
        write(run / "provenance.json", provenance)
        cache.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(
            PYTHONPATH=os.pathsep.join([str(harness / "hook"), str(source)]),
            TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
            TRITON_CACHE_DIR=str(cache / "triton"),
            TORCHINDUCTOR_COMPILE_THREADS="1",
            HIP_VISIBLE_DEVICES="0",
            HF_HUB_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            VLLM_LOGGING_COLOR="0",
            VLLM_LOGGING_LEVEL="INFO",
            QWEN_EXPERIMENT_RUN_DIR=str(run),
            QWEN_EXPERIMENT_DISABLE_PROMPT_CACHE="1",
            QWEN_EXPERIMENT_VARIANT=args.variant,
            QWEN_EXPERIMENT_INPUTS=str(inputs / "historical-16.json" if args.historical else inputs / "workloads.json"),
        )
        env["QWEN_EXPERIMENT_STAGE_BENCH"] = "1" if args.stage_bench else "0"
        env.pop("QWEN_EXPERIMENT_CAPTURE_DIR", None)
        env.pop("QWEN_EXPERIMENT_REPLAY_DIR", None)
        if args.diagnostic:
            env["QWEN_EXPERIMENT_CAPTURE_DIR"] = str(run / "captures")
        if args.replay_dir:
            env["QWEN_EXPERIMENT_REPLAY_DIR"] = str(Path(args.replay_dir).resolve())
        probe = subprocess.run(
            [sys.executable, "-c", 'import json,vllm_omni; print(json.dumps({"module":list(vllm_omni.__path__)}))'],
            cwd=source,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        write(run / "import-probe.json", {"stdout": probe.stdout, "stderr": probe.stderr})
        paths = json.loads(probe.stdout.strip().splitlines()[-1])["module"]
        if str(source / "vllm_omni") not in paths:
            raise RuntimeError(f"wrong import source: {paths}")
        command = [
            sys.executable,
            "-m",
            "vllm_omni.entrypoints.cli.main",
            "serve",
            str(checkpoint),
            "--served-model-name",
            "Qwen/Qwen-Image-2.1",
            "--omni",
            "--host",
            "127.0.0.1",
            "--port",
            "8091",
            "--num-gpus",
            "1",
            "--dtype",
            "bfloat16",
            "--tensor-parallel-size",
            "1",
            "--ulysses-degree",
            "1",
            "--ring-degree",
            "1",
            "--diffusion-attention-backend",
            "TORCH_SDPA",
            "--step-execution",
            "--max-num-seqs",
            "8" if args.historical else "1",
            "--vae-use-tiling",
        ]
        command += (
            ["--enforce-eager", "--no-enable-cuda-graph-decode"]
            if args.mode == "eager"
            else ["--diffusion-compile-granularity", "regional", "--enable-cuda-graph-decode"]
        )
        if args.variant == "O1":
            command += ["--diffusion-offload-config", '{"mode":"layer","components":["text_encoder"]}']
        write(
            run / "launch.json",
            {
                "command": command,
                "environment": {
                    k: v
                    for k, v in env.items()
                    if k.startswith(("QWEN_EXPERIMENT_", "TORCHINDUCTOR_", "TRITON_", "VLLM_", "MIOPEN_", "AITER_"))
                    or k in {"PYTHONPATH", "HIP_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TOKENIZERS_PARALLELISM"}
                },
            },
        )
        mark("load_start")
        with (run / "server.log").open("w") as log:
            server = subprocess.Popen(
                command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        write(run / "server.json", {"pid": server.pid})
        monitor = threading.Thread(target=telemetry, daemon=True)
        monitor.start()
        import requests

        deadline = time.monotonic() + args.startup_timeout
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode}")
            try:
                if requests.get("http://127.0.0.1:8091/health", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("server readiness timeout")
            time.sleep(1)
        mark("ready")
        if args.diagnostic:
            (run / "enable-capture").touch()
        doc = json.loads((inputs / ("historical-16.json" if args.historical else "workloads.json")).read_text())
        assets = {x["id"]: x for x in json.loads((inputs / "workloads.json").read_text()).get("assets", [])}
        cases = doc["cases"] if args.historical else [c for c in doc["cases"] if c["id"] in args.cases]
        if not cases or (not args.historical and {c["id"] for c in cases} != set(args.cases)):
            raise ValueError("unknown case ID")
        results = []

        def execute(case, phase, index):
            rid = f"{args.variant}-{phase}-{case['id']}-{index}-{uuid.uuid4().hex[:8]}"
            return request(case, assets, source, run / "requests" / rid, rid, phase)

        phases = (
            [("diagnostic", 1)]
            if args.diagnostic
            else [
                ("warmup", 32 if args.historical else args.warmups),
                ("measured", 1 if args.historical else args.samples),
            ]
        )
        for phase, count in phases:
            mark(phase + "_start")
            jobs = (
                [(cases[i % len(cases)], i) for i in range(count)]
                if args.historical and phase == "warmup"
                else [(case, i) for case in cases for i in range(count)]
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=8 if args.historical and not args.diagnostic else 1
            ) as pool:
                futures = [pool.submit(execute, case, phase, i) for case, i in jobs]
                for future in futures:
                    result = future.result()
                    results.append(result)
                    write(run / "samples.json", results)
                    if not result["success"]:
                        raise RuntimeError(f"request failed: {result}")
            mark(phase + "_end")
        write(
            run / "status.json",
            {
                "status": "completed",
                "diagnostic": args.diagnostic,
                "timing_claim": not args.diagnostic,
                "note": "Activation and numerical comparison require separate qualification.",
            },
        )
        workers = list(run.glob("worker-provenance-*.json"))
        if not workers:
            raise RuntimeError("No actual model-worker provenance marker")
        for path in workers:
            worker = json.loads(path.read_text())
            if not worker["native_prompt_cache_disabled"]:
                raise RuntimeError("Native prompt cache was not disabled")
            if args.mode == "compiled" and worker["compiled_blocks"] == 0:
                raise RuntimeError("Production blocks were not compiled")
    except Exception as exc:
        write(run / "status.json", {"status": "failed", "error": repr(exc)})
        raise
    finally:
        mark("shutdown_start")
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        stop.set()
        if monitor:
            monitor.join(timeout=20)
        if server is not None:
            write(run / "server-exit.json", {"returncode": server.returncode})
        mark("finished")


if __name__ == "__main__":
    main()
