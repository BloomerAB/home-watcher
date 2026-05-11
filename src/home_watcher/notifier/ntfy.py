"""NTFY push notification client.

NTFY API: POST to https://ntfy.sh/<topic>. Headers control title/priority/etc.
Body can be either text (message) or binary (attachment). For attachment +
message in same notification: send body=binary, message via "Message" header.

Non-ASCII header values (Swedish chars like ä, å, ö in titles) are encoded
as RFC 2047 MIME encoded-words so they survive httpx's strict ASCII-only
header validation and are decoded properly by NTFY clients.
"""

from __future__ import annotations

from email.header import Header

import httpx
import structlog

log = structlog.get_logger(__name__)


def _encode_header(value: str) -> str:
    """Encode header value safely for non-ASCII (RFC 2047 if needed).

    Header.encode() folds long values with CRLF + space, but HTTP headers
    can't contain newlines mid-value. We pass maxlinelen=10000 to disable
    folding so the entire encoded value stays on one line.

    Also strips any newlines from the input (e.g. multi-line messages we'd
    rather put in the body than a header).
    """
    flat = value.replace("\n", " ").replace("\r", " ")
    try:
        flat.encode("ascii")
        return flat
    except UnicodeEncodeError:
        return Header(flat, charset="utf-8").encode(maxlinelen=10000)


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
        click_url: str | None = None,
    ) -> bool:
        """Send a single NTFY notification with optional image attachment.

        When `image_bytes` is provided, the body is the image and `message` is
        carried in the X-Message header (NTFY renders this as the notification
        text while showing the image inline).
        """
        url = f"{self.base_url}/{self.topic}"
        headers = dict(self._headers)
        headers["X-Title"] = _encode_header(title)
        headers["X-Priority"] = str(priority)
        if tags:
            headers["X-Tags"] = _encode_header(",".join(tags))
        if click_url:
            headers["X-Click"] = click_url

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if image_bytes:
                    headers["X-Message"] = _encode_header(message)
                    headers["X-Filename"] = "snapshot.jpg"
                    resp = await client.post(url, headers=headers, content=image_bytes)
                else:
                    resp = await client.post(
                        url, headers=headers, content=message.encode("utf-8")
                    )
                resp.raise_for_status()
                return True
        except httpx.HTTPError as exc:
            log.error("ntfy_send_failed", error=str(exc), title=title)
            return False
