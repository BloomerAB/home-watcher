"""Poll Protect's REST events API instead of WebSocket.

Simpler and more reliable than the binary WS protocol. Yields ProtectUpdate-
shaped objects so the main event handler doesn't care about the source.

Polling cadence: every `interval_seconds` (default 3s). Each poll asks for
events since the previous poll's `now`. New events are yielded immediately.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import structlog

from .client import ProtectClient
from .events import ProtectUpdate

log = structlog.get_logger(__name__)


class ProtectEventPoller:
    def __init__(self, protect: ProtectClient, interval_seconds: float = 3.0) -> None:
        self.protect = protect
        self.interval = interval_seconds
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def stream(self) -> AsyncIterator[ProtectUpdate]:
        """Yield ProtectUpdate-like objects for each new Protect event."""
        # First poll: from "now" — only future events
        last_check_ms = int(time.time() * 1000)
        # Bootstrap so camera names are populated
        await self.protect.bootstrap()
        log.info("event_poller_started", interval=self.interval)

        while not self._stop.is_set():
            try:
                async for update in self._poll_once(last_check_ms):
                    yield update
                last_check_ms = int(time.time() * 1000)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("poll_error", error=str(exc))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except TimeoutError:
                pass

    async def _poll_once(self, since_ms: int) -> AsyncIterator[ProtectUpdate]:
        now_ms = int(time.time() * 1000)
        client = await self.protect._ensure_client()  # noqa: SLF001
        params = {"start": since_ms, "end": now_ms, "limit": 100}
        resp = await client.get(
            "/proxy/protect/api/events",
            params=params,
            headers=self.protect._auth_headers(),  # noqa: SLF001
        )
        if resp.status_code in (401, 403):
            await self.protect.login()
            resp = await client.get(
                "/proxy/protect/api/events",
                params=params,
                headers=self.protect._auth_headers(),  # noqa: SLF001
            )
        resp.raise_for_status()
        events: list[dict[str, Any]] = resp.json() or []

        for ev in events:
            update = self._to_update(ev)
            if update:
                yield update

    @staticmethod
    def _to_update(event: dict[str, Any]) -> ProtectUpdate | None:
        camera_id = event.get("camera")
        if not camera_id:
            return None
        ev_type = event.get("type", "")
        smart_types = event.get("smartDetectTypes") or []
        data: dict[str, Any] = {"_event_type": ev_type}
        if ev_type == "motion":
            data["lastMotion"] = event.get("start")
            data["isMotionDetected"] = True
        elif ev_type == "smartDetectZone":
            data["lastSmartDetect"] = {
                "smartDetectTypes": list(smart_types),
            }
            data["smartDetectTypes"] = list(smart_types)
        else:
            return None  # ignore other event types (ring, access, etc.)
        return ProtectUpdate(
            action="update",
            id=str(camera_id),
            model_key="camera",
            data=data,
        )
