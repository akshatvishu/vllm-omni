# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for bounded image response execution and app lifecycle ownership."""

import asyncio
import contextvars
import gc
import threading
import weakref
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from PIL import Image

from vllm_omni.entrypoints.openai.image_api_utils import (
    encode_image_base64,
    encode_image_base64_with_compression,
)
from vllm_omni.entrypoints.openai.images.response_executor import (
    ImageResponseExecutor,
    install_image_response_executor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=5))


async def event(event):
    assert await asyncio.to_thread(event.wait, 5), "worker did not reach checkpoint"


def block(started, release, value=None):
    started.set()
    assert release.wait(5), "worker was not released"
    return value


@pytest.mark.parametrize("fmt,quality", [("png", 100), ("png", 23), ("jpeg", 91), ("jpg", 50)])
def test_exact_encoded_bytes_and_order(fmt, quality):
    images = [Image.new("RGBA", (17 + i, 13), (i * 67, 120, 255 - i * 50, 50 + i * 70)) for i in range(3)]
    encode = encode_image_base64_with_compression
    expected = [encode(im, fmt, quality) for im in images]
    assert len(set(expected)) == 3

    async def check():
        pool = ImageResponseExecutor()
        try:
            actual = await pool.run(lambda: [encode(im, fmt, quality) for im in images])
            assert actual == expected
            assert await pool.run(encode_image_base64, images[0]) == encode_image_base64(images[0])
        finally:
            await pool.aclose()

    run(check())


def test_capacity_survives_active_and_queued_cancellation(monkeypatch):
    async def check():
        pool = ImageResponseExecutor()
        started = [threading.Event(), threading.Event()]
        release = threading.Event()
        queued_executed = threading.Event()
        lock = threading.Lock()
        unfinished = 0
        peak = 0
        submits = 0
        original = pool._executor.submit

        def submit(*args, **kwargs):
            nonlocal unfinished, peak, submits
            with lock:
                unfinished += 1
                submits += 1
                peak = max(peak, unfinished)
            future = original(*args, **kwargs)

            def done(_):
                nonlocal unfinished
                with lock:
                    unfinished -= 1

            future.add_done_callback(done)
            return future

        monkeypatch.setattr(pool._executor, "submit", submit)
        tasks = [asyncio.create_task(pool.run(block, s, release)) for s in started]
        try:
            for s in started:
                await event(s)
            queued = asyncio.create_task(pool.run(queued_executed.set))
            await asyncio.sleep(0)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            survivors = [asyncio.create_task(pool.run(lambda n=n: n)) for n in range(8)]
            await asyncio.sleep(0)
            assert submits == 2, "cancelled running waiters released submission capacity"
            assert not queued_executed.is_set()
            release.set()
            assert await asyncio.gather(*survivors) == list(range(8))
            assert peak <= 2
            assert not queued_executed.is_set()
        finally:
            release.set()
            await pool.aclose()

    run(check())


def test_cancelled_waiter_retains_input_until_worker_finishes():
    class Input:
        pass

    async def check():
        pool = ImageResponseExecutor()
        started, release = threading.Event(), threading.Event()
        payload = Input()
        ref = weakref.ref(payload)

        def worker(value):
            block(started, release)
            assert ref() is value

        task = asyncio.create_task(pool.run(worker, payload))
        del payload
        try:
            await event(started)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            gc.collect()
            assert ref() is not None
        finally:
            release.set()
            await pool.aclose()
        del task
        gc.collect()
        assert ref() is None

    run(check())


def test_contextvars_and_exception_propagation():
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("request")

    async def check():
        pool = ImageResponseExecutor()
        marker.set("request-123")

        def worker(*, suffix):
            value = marker.get()
            marker.set("worker-only")
            return value + suffix, threading.get_ident()

        try:
            result, ident = await pool.run(worker, suffix="!")
            assert result == "request-123!"
            assert ident != threading.get_ident()
            assert marker.get() == "request-123"
            error = ValueError("encoder failure")

            def fail():
                raise error

            with pytest.raises(ValueError) as caught:
                await pool.run(fail)
            assert caught.value is error
            assert await pool.run(lambda: 42) == 42
        finally:
            await pool.aclose()

    run(check())


def test_cancelled_waiter_worker_exception_is_retrieved():
    async def check():
        loop = asyncio.get_running_loop()
        errors = []
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        pool = ImageResponseExecutor(max_workers=1)
        started, release = threading.Event(), threading.Event()

        def fail():
            block(started, release)
            raise ValueError("disconnected encoder")

        task = asyncio.create_task(pool.run(fail))
        try:
            await event(started)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
            # Wait for failure completion without shutdown gathering its exception.
            assert await pool.run(lambda: "after failure") == "after failure"
            del task
            gc.collect()
            await asyncio.sleep(0)
            assert not errors
        finally:
            release.set()
            await pool.aclose()
            loop.set_exception_handler(previous)

    run(check())


def test_submission_failure_restores_capacity(monkeypatch):
    async def check():
        pool = ImageResponseExecutor(max_workers=1)
        original = pool._executor.submit

        def fail(*args, **kwargs):
            raise RuntimeError("submit failed")

        try:
            monkeypatch.setattr(pool._executor, "submit", fail)
            with pytest.raises(RuntimeError, match="submit failed"):
                await pool.run(lambda: 1)
            monkeypatch.setattr(pool._executor, "submit", original)
            assert await pool.run(lambda: 2) == 2
        finally:
            await pool.aclose()

    run(check())


def test_shutdown_drains_rejects_jobs_and_is_idempotent():
    async def check():
        pool = ImageResponseExecutor(max_workers=1)
        started, release = threading.Event(), threading.Event()
        queued_executed = threading.Event()
        active = asyncio.create_task(pool.run(block, started, release, "done"))
        try:
            await event(started)
            queued = asyncio.create_task(pool.run(queued_executed.set))
            await asyncio.sleep(0)
            close = asyncio.create_task(pool.aclose())
            await asyncio.sleep(0)
            assert not close.done()
            with pytest.raises(RuntimeError, match="closed"):
                await pool.run(lambda: None)
            close.cancel()
            await asyncio.sleep(0)
            assert not close.done()
            close.cancel()
            await asyncio.sleep(0)
            assert not close.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await close
            assert not any(t.is_alive() for t in pool._executor._threads)
            assert await active == "done"
            with pytest.raises(RuntimeError, match="closed"):
                await queued
            await asyncio.gather(pool.aclose(), pool.aclose())
            await pool.aclose()
            assert not queued_executed.is_set()
            assert not any(t.is_alive() for t in pool._executor._threads)
        finally:
            release.set()
            await pool.aclose()

    run(check())


def test_lifespan_exception_keeps_original_context_and_cleans_pool():
    async def check():
        visits = []
        state = {"original": object()}

        @asynccontextmanager
        async def original(app):
            visits.append("enter")
            try:
                yield state
            finally:
                visits.append("exit")

        app = FastAPI(lifespan=original)
        install_image_response_executor(app)
        with pytest.raises(ValueError, match="endpoint body"):
            async with app.router.lifespan_context(app) as actual:
                assert actual is state
                pool = app.state.image_response_executor
                assert await pool.run(lambda: 7) == 7
                raise ValueError("endpoint body")
        assert visits == ["enter", "exit"]
        assert not any(t.is_alive() for t in pool._executor._threads)
        with pytest.raises(RuntimeError, match="closed"):
            await pool.run(lambda: 1)

    run(check())


def test_fastapi_owned_pool_endpoint():
    app = FastAPI()
    install_image_response_executor(app)
    image = Image.new("RGB", (8, 9), (12, 34, 56))
    encode = encode_image_base64

    @app.get("/image")
    async def endpoint(request: Request):
        return {
            "images": await request.app.state.image_response_executor.run(
                lambda: [encode(image), encode(image.transpose(Image.Transpose.ROTATE_90))]
            )
        }

    with TestClient(app) as client:
        response = client.get("/image")
        assert response.status_code == 200
        assert response.json() == {"images": [encode(image), encode(image.transpose(Image.Transpose.ROTATE_90))]}
        pool = app.state.image_response_executor
    assert not any(t.is_alive() for t in pool._executor._threads)
