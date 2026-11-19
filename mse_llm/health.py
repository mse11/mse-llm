"""Async readiness probes and small HTTP helpers (§3.3). Replaces the bash `curl -sf /health` loop."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx


class NotReady(RuntimeError):
    pass


def root_of(base_url: str) -> str:
    """http://host:port/v1 -> http://host:port"""
    u = base_url.rstrip("/")
    return u[:-3] if u.endswith("/v1") else u


async def wait_ready(url: str, timeout: float, *, interval: float = 0.5,
                     exited: Callable[[], int | None] | None = None,
                     progress: Callable[[float, str], None] | None = None) -> float:
    """Poll `url` until it answers 200 (llama-server /health answers 503 while it loads the model).
    Fails fast when `exited()` returns an exit code. Returns the seconds it took."""
    t0 = time.monotonic()
    last = "no answer"
    async with httpx.AsyncClient(timeout=2.0) as client:
        while True:
            rc = exited() if exited else None
            if rc is not None:
                raise NotReady(f"process exited with code {rc} before {url} was ready")
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    return time.monotonic() - t0
                last = f"HTTP {r.status_code}"
                try:
                    msg = r.json().get("error", {}).get("message")
                    if msg:
                        last += f" {msg}"
                except (ValueError, AttributeError):
                    pass
            except httpx.HTTPError as e:
                last = type(e).__name__
            el = time.monotonic() - t0
            if progress:
                progress(el, last)
            if el >= timeout:
                raise NotReady(f"{url} not ready within {timeout:.0f} s (last: {last})")
            await asyncio.sleep(interval)


async def get_json(url: str, timeout: float = 5.0) -> Any:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()


async def post_json(url: str, payload: dict[str, Any], timeout: float = 30.0) -> tuple[int, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, json=payload)
        try:
            body = r.json()
        except ValueError:
            body = r.text
        return r.status_code, body


async def sse_events(url: str) -> AsyncIterator[dict[str, Any]]:
    """Server-sent events as dicts (the `data:` lines, JSON-decoded)."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=None)) as client:
        async with client.stream("GET", url) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                try:
                    yield json.loads(data)
                except ValueError:
                    yield {"raw": data}
