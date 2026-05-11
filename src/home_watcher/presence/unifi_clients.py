"""Query UniFi Network controller for currently-connected client MACs.

Used to determine family presence: if any of the configured family-member
phone MACs appear in the active client list, family is "home".
"""

import time
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)

CACHE_TTL_SECONDS = 60


class UnifiClientsLookup:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        family_macs: dict[str, str],
        verify_tls: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.family_macs = {mac.lower(): name for mac, name in family_macs.items()}
        self.verify_tls = verify_tls
        self._client: httpx.AsyncClient | None = None
        self._csrf: str | None = None
        self._cache: tuple[float, set[str]] | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                verify=self.verify_tls,
                timeout=10.0,
                cookies=httpx.Cookies(),
            )
        return self._client

    async def _login(self) -> None:
        client = await self._ensure_client()
        resp = await client.post(
            "/api/auth/login",
            json={"username": self.username, "password": self.password},
        )
        resp.raise_for_status()
        self._csrf = resp.headers.get("x-csrf-token")

    async def _list_active_macs(self) -> set[str]:
        client = await self._ensure_client()
        headers = {"X-CSRF-Token": self._csrf} if self._csrf else {}
        try:
            resp = await client.get(
                "/proxy/network/api/s/default/stat/sta", headers=headers
            )
            if resp.status_code in (401, 403):
                await self._login()
                resp = await client.get(
                    "/proxy/network/api/s/default/stat/sta",
                    headers={"X-CSRF-Token": self._csrf or ""},
                )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except httpx.HTTPError as exc:
            log.warning("unifi_clients_query_failed", error=str(exc))
            return set()
        return {str(c.get("mac", "")).lower() for c in data.get("data", []) if c.get("mac")}

    async def family_at_home(self) -> bool:
        now = time.monotonic()
        if self._cache and now - self._cache[0] < CACHE_TTL_SECONDS:
            active = self._cache[1]
        else:
            active = await self._list_active_macs()
            self._cache = (now, active)
        return any(mac in active for mac in self.family_macs)

    async def family_members_at_home(self) -> list[str]:
        now = time.monotonic()
        if self._cache and now - self._cache[0] < CACHE_TTL_SECONDS:
            active = self._cache[1]
        else:
            active = await self._list_active_macs()
            self._cache = (now, active)
        return [name for mac, name in self.family_macs.items() if mac in active]

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
