"""Decoder for UniFi Protect WebSocket binary update protocol.

Packet structure (from hjdhjd/unifi-protect reverse engineering):
  [8-byte action header][action payload]
  [8-byte data header][data payload]

Header layout (8 bytes):
  byte 0: packet type (1=action, 2=payload)
  byte 1: payload format (1=JSON, 2=UTF8, 3=Buffer)
  byte 2: compressed (0=raw, 1=zlib)
  byte 3: reserved
  bytes 4-7: payload size (uint32 big-endian)

Action payload (JSON): {action: "add"|"update", id, modelKey, newUpdateId}
Data payload: depends on format flag.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from typing import Any

HEADER_SIZE = 8

PACKET_TYPE_ACTION = 1
PACKET_TYPE_PAYLOAD = 2

FORMAT_JSON = 1
FORMAT_UTF8 = 2
FORMAT_BUFFER = 3


@dataclass
class ProtectUpdate:
    action: str
    """'add' or 'update'."""
    id: str
    """Device or event ID."""
    model_key: str
    """e.g. 'camera', 'event', 'bridge'."""
    data: dict[str, Any]
    """Partial state delta (only changed fields)."""


def decode(packet: bytes) -> ProtectUpdate | None:
    if len(packet) < HEADER_SIZE * 2:
        return None

    action_header = _parse_header(packet[:HEADER_SIZE])
    if action_header is None:
        return None
    action_end = HEADER_SIZE + action_header["size"]
    if len(packet) < action_end + HEADER_SIZE:
        return None

    action_bytes = _decode_payload(
        packet[HEADER_SIZE:action_end],
        format_flag=action_header["format"],
        compressed=action_header["compressed"],
    )
    action_data = json.loads(action_bytes)

    data_header = _parse_header(packet[action_end : action_end + HEADER_SIZE])
    if data_header is None:
        return None
    data_start = action_end + HEADER_SIZE
    data_end = data_start + data_header["size"]
    if len(packet) < data_end:
        return None

    data_bytes = _decode_payload(
        packet[data_start:data_end],
        format_flag=data_header["format"],
        compressed=data_header["compressed"],
    )
    data_obj: dict[str, Any] = json.loads(data_bytes) if data_header["format"] == FORMAT_JSON else {}

    return ProtectUpdate(
        action=action_data.get("action", ""),
        id=action_data.get("id", ""),
        model_key=action_data.get("modelKey", ""),
        data=data_obj,
    )


def _parse_header(buf: bytes) -> dict[str, int] | None:
    if len(buf) < HEADER_SIZE:
        return None
    packet_type, fmt, compressed, _reserved, size = struct.unpack(">BBBBI", buf)
    return {
        "type": packet_type,
        "format": fmt,
        "compressed": compressed,
        "size": size,
    }


def _decode_payload(buf: bytes, *, format_flag: int, compressed: int) -> bytes:
    if compressed == 1:
        buf = zlib.decompress(buf)
    return buf


def is_motion_event(update: ProtectUpdate) -> bool:
    """True if this is a camera update indicating motion or smart detect."""
    if update.model_key != "camera" or update.action != "update":
        return False
    return any(key in update.data for key in ("lastMotion", "lastSmartDetect", "isMotionDetected"))


def smart_detect_types(update: ProtectUpdate) -> list[str]:
    """Extract list of smart detect types (e.g. ['person'], ['vehicle'])."""
    last_sd = update.data.get("lastSmartDetect")
    if isinstance(last_sd, dict):
        types = last_sd.get("smartDetectTypes")
        if isinstance(types, list):
            return [str(t) for t in types]
    raw = update.data.get("smartDetectTypes")
    if isinstance(raw, list):
        return [str(t) for t in raw]
    return []
