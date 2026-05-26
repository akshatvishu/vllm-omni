# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def get_transfer_extra_config(transfer_manager: Any) -> dict[str, Any]:
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    if isinstance(raw_cfg, dict):
        return raw_cfg.get("extra", raw_cfg) or {}
    return {}


def get_chunk_config_int(
    cfg: dict[str, Any],
    key: str,
    fallback: int,
    *,
    warning_prefix: str | None = None,
) -> int:
    if key not in cfg:
        if warning_prefix is not None:
            logger.warning("%s missing %s, using fallback value %s", warning_prefix, key, fallback)
        return fallback
    return int(cfg[key])


def get_request_payload_store(transfer_manager: Any) -> dict[str, Any]:
    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload
    return request_payload
