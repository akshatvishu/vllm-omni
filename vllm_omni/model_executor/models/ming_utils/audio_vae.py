# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn


class StreamingLinearUpsample(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode="linear", align_corners=False)

    def forward(self, x, state=None, is_last=False):
        if x is None and is_last and (state is None or state.get("prev_chunk") is None):
            raise ValueError("Received end-of-stream without any latent chunk to upsample.")
        if state is None:
            state = {"prev_chunk": None, "history_last": None, "is_first": True}

        if x is None and not is_last:
            return None, state

        if state["is_first"] and is_last:
            out = self.upsampler(x.transpose(1, 2)).transpose(1, 2)
            return out, None

        output_chunks = []

        if state["is_first"]:
            state["prev_chunk"] = x
            state["is_first"] = False
            if not is_last:
                return None, state

        if state["prev_chunk"] is not None:
            p = state["prev_chunk"].transpose(1, 2)

            if state["history_last"] is None:
                lookahead = x[:, :1, :].transpose(1, 2)
                inp = torch.cat([p, lookahead], dim=2)
                up = self.upsampler(inp)
                out_prev = up[:, :, : p.size(2) * self.scale_factor]
            else:
                lookahead = x[:, :1, :].transpose(1, 2)
                inp = torch.cat([state["history_last"], p, lookahead], dim=2)
                up = self.upsampler(inp)
                start = self.scale_factor
                end = start + p.size(2) * self.scale_factor
                out_prev = up[:, :, start:end]

            output_chunks.append(out_prev.transpose(1, 2))
            state["history_last"] = p[:, :, -1:]
            state["prev_chunk"] = x

        if is_last:
            p = state["prev_chunk"].transpose(1, 2)
            inp = torch.cat([state["history_last"], p], dim=2)
            up = self.upsampler(inp)
            out_last = up[:, :, self.scale_factor :]
            output_chunks.append(out_last.transpose(1, 2))
            state = None

        final_out = torch.cat(output_chunks, dim=1) if output_chunks else None
        return final_out, state
