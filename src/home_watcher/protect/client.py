"""UniFi Protect REST client: auth, bootstrap, snapshot fetch."""

from __future__ import annotations

from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)


class ProtectClient:
    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        verify_tls: bool = False,
    ) -> None:
        self.base_url = f"https://{host}"
        self.username = username
        self.password = password
        self.verify_tls = verify_tls
        self._client: httpx.AsyncClient | None = None
        self._csrf: str | None = None
        self._cameras_by_id: dict[str, dict[str, Any]] = {}
        self._last_update_id: str = ""

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                verify=self.verify_tls,
                timeout=15.0,
                cookies=httpx.Cookies(),
            )
        return self._client

    async def login(self) -> None:
        client = await self._ensure_client()
        resp = await client.post(
            "/api/auth/login",
            json={"username": self.username, "password": self.password},
        )
        resp.raise_for_status()
        self._csrf = resp.headers.get("x-csrf-token")
        log.info("protect_login_ok", csrf_present=bool(self._csrf))

    def _auth_headers(self) -> dict[str, str]:
        return {"X-CSRF-Token": self._csrf} if self._csrf else {}

    async def bootstrap(self) -> dict[str, Any]:
        """Fetch bootstrap, populate camera cache, return raw bootstrap dict."""
        client = await self._ensure_client()
        resp = await client.get("/proxy/protect/api/bootstrap", headers=self._auth_headers())
        if resp.status_code in (401, 403):
            await self.login()
            resp = await client.get("/proxy/protect/api/bootstrap", headers=self._auth_headers())
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        self._last_update_id = str(data.get("lastUpdateId", ""))
        cameras = data.get("cameras") or []
        self._cameras_by_id = {str(c["id"]): c for c in cameras if "id" in c}
        log.info("protect_bootstrap_ok", cameras=len(self._cameras_by_id))
        return data

    @property
    def last_update_id(self) -> str:
        return self._last_update_id

    def camera_name(self, camera_id: str) -> str:
        cam = self._cameras_by_id.get(camera_id)
        if not cam:
            return camera_id
        return str(cam.get("name", camera_id))

    async def list_events(
        self,
        start_ms: int,
        end_ms: int,
        *,
        types: list[str] | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Fetch historical events from Protect REST API."""
        client = await self._ensure_client()
        params: dict[str, Any] = {"start": start_ms, "end": end_ms, "limit": limit}
        if types:
            params["types"] = ",".join(types)
        resp = await client.get(
            "/proxy/protect/api/events",
            params=params,
            headers=self._auth_headers(),
        )
        if resp.status_code in (401, 403):
            await self.login()
            resp = await client.get(
                "/proxy/protect/api/events",
                params=params,
                headers=self._auth_headers(),
            )
        resp.raise_for_status()
        return resp.json() or []

    async def fetch_event_thumbnail(self, event_id: str) -> bytes | None:
        """Fetch the still JPEG thumbnail captured at the event's moment."""
        client = await self._ensure_client()
        try:
            resp = await client.get(
                f"/proxy/protect/api/events/{event_id}/thumbnail",
                headers=self._auth_headers(),
                timeout=15.0,
            )
            if resp.status_code in (401, 403):
                await self.login()
                resp = await client.get(
                    f"/proxy/protect/api/events/{event_id}/thumbnail",
                    headers=self._auth_headers(),
                    timeout=15.0,
                )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPError as exc:
            log.warning("event_thumbnail_failed", event_id=event_id, error=str(exc))
            return None

    async def fetch_snapshot(self, camera_id: str, width: int = 1920) -> bytes | None:
        client = await self._ensure_client()
        try:
            resp = await client.get(
                f"/proxy/protect/api/cameras/{camera_id}/snapshot",
                params={"force": "true", "w": width},
                headers=self._auth_headers(),
                timeout=10.0,
            )
            if resp.status_code in (401, 403):
                await self.login()
                resp = await client.get(
                    f"/proxy/protect/api/cameras/{camera_id}/snapshot",
                    params={"force": "true", "w": width},
                    headers=self._auth_headers(),
                    timeout=10.0,
                )
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPError as exc:
            log.warning("snapshot_fetch_failed", camera_id=camera_id, error=str(exc))
            return None

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
