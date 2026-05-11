"""NTFY push notification client."""

import httpx
import structlog

log = structlog.get_logger(__name__)


class NtfyNotifier:
    def __init__(self, base_url: str, topic: str, token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.topic = topic
        self._headers: dict[str, str] = {}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"

    async def send(
        self,
        title: str,
        message: str,
        *,
        priority: int = 3,
        tags: list[str] | None = None,
        image_bytes: bytes | None = None,
    ) -> bool:
        url = f"{self.base_url}/{self.topic}"
        headers = dict(self._headers)
        headers["Title"] = title
        headers["Priority"] = str(priority)
        if tags:
            headers["Tags"] = ",".join(tags)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if image_bytes:
                    headers["Filename"] = "snapshot.jpg"
                    resp = await client.put(
                        url, headers=headers, content=image_bytes
                    )
                    if resp.is_success:
                        await client.post(
                            url, headers=self._headers, content=message.encode("utf-8")
                        )
                else:
                    resp = await client.post(
                        url, headers=headers, content=message.encode("utf-8")
                    )
                resp.raise_for_status()
                return True
        except httpx.HTTPError as exc:
            log.error("ntfy_send_failed", error=str(exc), title=title)
            return False
