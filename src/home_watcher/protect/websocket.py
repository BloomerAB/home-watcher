"""UniFi Protect WebSocket realtime event subscriber.

Connects to /proxy/protect/ws/updates with the bootstrap's lastUpdateId,
yields decoded ProtectUpdate objects. Reconnects with exponential backoff
on disconnect.
"""

from __future__ import annotations

import asyncio
import ssl
from collections.abc import AsyncIterator
from contextlib import suppress

import structlog
import websockets
from websockets.asyncio.client import ClientConnection

from .client import ProtectClient
from .events import ProtectUpdate, decode

log = structlog.get_logger(__name__)

MIN_BACKOFF_SECONDS = 1
MAX_BACKOFF_SECONDS = 60


class ProtectWebSocket:
    def __init__(self, protect: ProtectClient) -> None:
        self.protect = protect
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def stream(self) -> AsyncIterator[ProtectUpdate]:
        """Yield updates indefinitely, reconnecting on failure."""
        backoff = MIN_BACKOFF_SECONDS
        while not self._stop.is_set():
            try:
                await self.protect.bootstrap()
                last_id = self.protect.last_update_id
                async for update in self._connect_and_consume(last_id):
                    backoff = MIN_BACKOFF_SECONDS  # reset on first message
                    yield update
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("ws_loop_error", error=str(exc), backoff=backoff)
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)

    async def _connect_and_consume(self, last_update_id: str) -> AsyncIterator[ProtectUpdate]:
        host = self.protect.base_url.replace("https://", "")
        ws_url = f"wss://{host}/proxy/protect/ws/updates?lastUpdateId={last_update_id}"

        cookies = self.protect._client.cookies if self.protect._client else None  # noqa: SLF001
        if cookies is None:
            raise RuntimeError("ProtectClient not initialized (call login/bootstrap first)")

        cookie_header = "; ".join(f"{name}={value}" for name, value in cookies.items())
        headers: list[tuple[str, str]] = [("Cookie", cookie_header)]

        ssl_ctx = ssl.create_default_context()
        if not self.protect.verify_tls:
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE

        log.info("ws_connecting", url=ws_url)
        async with websockets.connect(
            ws_url,
            additional_headers=headers,
            ssl=ssl_ctx,
            max_size=2**24,
            ping_interval=30,
            ping_timeout=10,
        ) as ws:
            log.info("ws_connected")
            async for raw in self._iter_messages(ws):
                if not isinstance(raw, bytes):
                    continue
                update = decode(raw)
                if update is not None:
                    yield update

    async def _iter_messages(self, ws: ClientConnection) -> AsyncIterator[bytes | str]:
        stopper = asyncio.create_task(self._stop.wait())
        try:
            while not self._stop.is_set():
                recv = asyncio.create_task(ws.recv())
                done, pending = await asyncio.wait(
                    {recv, stopper}, return_when=asyncio.FIRST_COMPLETED
                )
                if stopper in done:
                    recv.cancel()
                    with suppress(asyncio.CancelledError):
                        await recv
                    return
                yield recv.result()
                for p in pending:
                    if p is not stopper:
                        p.cancel()
        finally:
            stopper.cancel()
            with suppress(asyncio.CancelledError):
                await stopper
