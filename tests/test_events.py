"""Protect WebSocket binary protocol decoder tests."""

from __future__ import annotations

import json
import struct

from home_watcher.protect.events import (
    PACKET_TYPE_ACTION,
    PACKET_TYPE_PAYLOAD,
    decode,
    event_camera_id,
    is_motion_event,
    smart_detect_types,
)


def _build_packet(action: dict[str, object], data: dict[str, object]) -> bytes:
    action_bytes = json.dumps(action).encode("utf-8")
    data_bytes = json.dumps(data).encode("utf-8")
    action_header = struct.pack(">BBBBI", PACKET_TYPE_ACTION, 1, 0, 0, len(action_bytes))
    data_header = struct.pack(">BBBBI", PACKET_TYPE_PAYLOAD, 1, 0, 0, len(data_bytes))
    return action_header + action_bytes + data_header + data_bytes


def test_decode_camera_motion_update() -> None:
    packet = _build_packet(
        {"action": "update", "id": "cam-abc", "modelKey": "camera"},
        {"lastMotion": 1715430000000, "isMotionDetected": True},
    )
    update = decode(packet)
    assert update is not None
    assert update.action == "update"
    assert update.id == "cam-abc"
    assert update.model_key == "camera"
    assert update.data["lastMotion"] == 1715430000000
    assert is_motion_event(update)


def test_decode_truncated_packet_returns_none() -> None:
    assert decode(b"\x01\x01\x00\x00\x00\x00\x00") is None


def test_is_motion_event_only_for_camera_updates() -> None:
    packet = _build_packet(
        {"action": "update", "id": "x", "modelKey": "bridge"},
        {"lastMotion": 1},
    )
    update = decode(packet)
    assert update is not None
    assert not is_motion_event(update)


def test_smart_detect_types_extraction() -> None:
    packet = _build_packet(
        {"action": "update", "id": "cam", "modelKey": "camera"},
        {"lastSmartDetect": {"smartDetectTypes": ["person", "vehicle"]}},
    )
    update = decode(packet)
    assert update is not None
    assert smart_detect_types(update) == ["person", "vehicle"]


def test_smart_detect_types_top_level() -> None:
    packet = _build_packet(
        {"action": "update", "id": "cam", "modelKey": "camera"},
        {"smartDetectTypes": ["animal"], "lastMotion": 1},
    )
    update = decode(packet)
    assert update is not None
    assert smart_detect_types(update) == ["animal"]


def test_event_add_motion_is_motion_event() -> None:
    packet = _build_packet(
        {"action": "add", "id": "evt-1", "modelKey": "event"},
        {"type": "motion", "camera": "cam-xyz", "start": 1234567890},
    )
    update = decode(packet)
    assert update is not None
    assert is_motion_event(update)
    assert event_camera_id(update) == "cam-xyz"


def test_event_add_smart_detect_zone_is_motion_event() -> None:
    packet = _build_packet(
        {"action": "add", "id": "evt-2", "modelKey": "event"},
        {"type": "smartDetectZone", "camera": "cam-abc",
         "smartDetectTypes": ["person", "vehicle"]},
    )
    update = decode(packet)
    assert update is not None
    assert is_motion_event(update)
    assert smart_detect_types(update) == ["person", "vehicle"]
    assert event_camera_id(update) == "cam-abc"


def test_event_unknown_type_ignored() -> None:
    packet = _build_packet(
        {"action": "add", "id": "evt-3", "modelKey": "event"},
        {"type": "ring", "camera": "doorbell-1"},
    )
    update = decode(packet)
    assert update is not None
    assert not is_motion_event(update)


def test_camera_update_camera_id() -> None:
    packet = _build_packet(
        {"action": "update", "id": "cam-zzz", "modelKey": "camera"},
        {"lastMotion": 1234},
    )
    update = decode(packet)
    assert update is not None
    assert event_camera_id(update) == "cam-zzz"
