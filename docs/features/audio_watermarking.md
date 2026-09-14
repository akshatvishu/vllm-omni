# Audio watermarking

Audio watermarking embeds an AudioSeal watermark in generated audio. It is disabled by default and applies to the whole engine when enabled.

## Install and enable

This branch depends on the streaming repair in [AudioSeal PR #106](https://github.com/facebookresearch/audioseal/pull/106). The `watermarking` extra pins commit `ab35ea6add4856f2a4ae0668230e6d69697d3f6a` from that PR. A released AudioSeal version has not yet replaced this development dependency. Python 3.10 or later is required for streaming.

Install the extra from the checked-out vLLM-Omni source in your existing environment:

```bash
.venv/bin/python -m pip install -e '.[watermarking]'
```

Add this argument to your audio model's normal serving command:

```bash
--watermark-config '{"audio":{"algorithm":"audioseal"}}'
```

For the Python API, pass `WatermarkConfig({"audio": {"algorithm": "audioseal"}})` as `watermark_config` when constructing `Omni`. Import `WatermarkConfig` from `vllm_omni.config.watermarking`.

Only `algorithm` is accepted in the audio configuration. Unsupported keys raise a configuration error. Message selection, watermark strength, and per-request enablement are not configurable. The adapter creates a deterministic message using a private CPU random generator, without reseeding the process's CPU or CUDA generators.

## Streaming behavior

AudioSeal runs on CPU, with access serialized for each stage pool. Each request retains its own encoder and decoder state. The adapter buffers fewer than one complete model frame between calls. For the pinned checkpoint, a frame contains 320 samples.

Output chunk sizes can differ from input chunk sizes, including an empty output while the first frame is incomplete. When the output processor completes the request, the adapter pads the final partial frame for watermark generation and removes that padding before returning audio. The total sample count stays unchanged. A terminal event without an audio payload still flushes the pending samples. Aborting a request discards its pending samples and model state.

The adapter preserves the original sample rate. For stereo audio, it watermarks the mono average and adds the watermark residual to both channels while limiting the amplitude to avoid clipping.

## Failures

Audio validation and model execution failures (`TypeError`, `ValueError`, or `RuntimeError`) log a warning and return the original generated audio. This also returns any original samples buffered from the preceding chunk. If final watermark generation fails, the original final samples are returned without padding. Broken watermark state is released, and a later chunk can start with fresh state. Other requests continue. Audio already delivered earlier in a streaming response cannot be changed.

Generated audio should include integer sample-rate metadata. Missing or invalid metadata prevents watermarking. If buffered samples cannot be combined safely with the current chunk because its layout or sample rate changed or is missing, or no output completion exists to deliver them, the request returns an error instead of silently losing audio. Malformed output payloads that are not arrays or tensors also remain request errors. Unexpected exception types propagate.

This best-effort behavior applies to the generated-output methods, `watermark_output()` and `finish_request_output()`. Direct calls to `watermark()` still raise validation and model errors. There is no strict-mode setting in this version.
