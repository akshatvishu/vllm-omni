# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from https://github.com/inclusionAI/Ming-omni-tts/blob/main/spkemb_extractor.py
import os

import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi


def resolve_model_to_local_path(model):
    if os.path.isdir(model):
        return model

    from huggingface_hub import snapshot_download

    return snapshot_download(model)


class MingSpeakerEmbeddingExtractor:
    def __init__(self, model, target_sr=16000):
        local_model_path = resolve_model_to_local_path(model)
        campplus_path = os.path.join(local_model_path, "campplus.onnx")
        if not os.path.exists(campplus_path):
            raise RuntimeError(f"Missing Ming speaker extractor model: {campplus_path}")

        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 2
        self.session = onnxruntime.InferenceSession(
            campplus_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.target_sr = int(target_sr)

    def extract_from_waveform(self, waveform, sample_rate):
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.as_tensor(waveform)

        tensor = waveform.detach().to(torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if int(sample_rate) != self.target_sr:
            tensor = torchaudio.transforms.Resample(orig_freq=int(sample_rate), new_freq=self.target_sr)(tensor)

        feat = kaldi.fbank(
            tensor,
            num_mel_bins=80,
            dither=0,
            sample_frequency=self.target_sr,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.session.run(
            None,
            {self.session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()},
        )[0].flatten()
        return torch.tensor(embedding, dtype=torch.float32)

    def extract_from_file(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        return self.extract_from_waveform(waveform, sample_rate)

    def extract_many(self, audio_paths):
        return [self.extract_from_file(path) for path in audio_paths]
