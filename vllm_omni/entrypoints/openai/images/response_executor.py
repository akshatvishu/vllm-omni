# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""App-owned worker pool for CPU image response construction."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from contextvars import copy_context
from functools import partial
from typing import TYPE_CHECKING, ParamSpec, TypeVar

if TYPE_CHECKING:
    from fastapi import FastAPI

P = ParamSpec("P")
T = TypeVar("T")


class ImageResponseExecutor:
    """Bound submitted jobs until completion, even if their request is cancelled."""

    def __init__(self, max_workers: int = 2) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="image-response")
        self._capacity = asyncio.Semaphore(max_workers)
        self._pending: set[asyncio.Future] = set()
        self._closed = False
        self._close_task: asyncio.Task | None = None

    async def run(self, func: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        if self._closed:
            raise RuntimeError("Image response executor is closed")
        await self._capacity.acquire()
        if self._closed:
            self._capacity.release()
            raise RuntimeError("Image response executor is closed")
        try:
            # Preserve request context and keep arguments alive in the worker.
            call = partial(copy_context().run, func, *args, **kwargs)
            future = asyncio.get_running_loop().run_in_executor(self._executor, call)
        except BaseException:
            self._capacity.release()
            raise
        self._pending.add(future)
        future.add_done_callback(self._finished)
        # Cancelling an HTTP waiter cannot stop its running codec thread.
        return await asyncio.shield(future)

    def _finished(self, future: asyncio.Future) -> None:
        self._pending.discard(future)
        self._capacity.release()
        # A disconnected request may no longer await a failing job.
        if not future.cancelled():
            future.exception()

    async def _drain(self) -> None:
        if self._pending:
            await asyncio.gather(*tuple(self._pending), return_exceptions=True)
        await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)

    async def aclose(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._drain())
        cancelled = False
        while not self._close_task.done():
            try:
                await asyncio.shield(self._close_task)
            except asyncio.CancelledError:
                # Finish ownership cleanup before lifespan teardown can close
                # the loop, even if shutdown itself is cancelled repeatedly.
                cancelled = True
        self._close_task.result()
        if cancelled:
            raise asyncio.CancelledError


def install_image_response_executor(app: FastAPI) -> None:
    """Tie pool ownership to ASGI lifespan, including exceptional exits."""
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with original_lifespan(app) as state:
            executor = ImageResponseExecutor(max_workers=2)
            app.state.image_response_executor = executor
            try:
                yield state
            finally:
                await executor.aclose()

    app.router.lifespan_context = lifespan
