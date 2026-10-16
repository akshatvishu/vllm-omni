# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import os

if os.environ.get("QWEN_SYNC_ARM"):
    from sync_overlay import install

    install()
