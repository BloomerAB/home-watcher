# home-watcher

Event-driven multi-signal home camera alerting via UniFi Protect.

## What this is

Subscribes to UniFi Protect's WebSocket events API. On each motion / smart-detection event:
1. Fetches a JPEG snapshot from the camera
2. Runs local face recognition (dlib) against a SQLite-backed face DB
3. Combines signals (face match, time of day, family-presence, per-camera weights, object type)
4. Decides: alert (push notif) or silent

No video streams, no Frigate, no Home Assistant. Single Python service in k3s.

## Why not Frigate?

UniFi Protect's RTSP/RTSPS streams are unreliable in production (camera reboots, firmware
updates, token rotation). The WebSocket events API is what the Protect app uses itself,
and is fundamentally more stable.

## Signals used (V1)

- Face recognition (dlib `face_recognition` library, SQLite store)
- Smart Detection categories from Protect (person/animal/vehicle/package)
- Per-camera always-alert rules (e.g., vehicles at Entrance)
- Time of day (night hours weight higher)
- Family presence (UniFi Network API — are family phones on Wi-Fi?)
- Per-camera alert weight (some zones are inherently more critical)

## Deferred to V2/V3

- Body re-identification (clothing color tracking)
- Custom pet classifier (YOLOv8 trained on user's pets)
- Motion pattern detection (approaching vs passing)
- Egen NTFY-server (V1 uses public ntfy.sh)

## Development

```bash
uv venv --python 3.13
uv pip install -e ".[dev]"
pytest
```

## Configuration

Env vars (see `src/home_watcher/config.py`):

- `UNIFI_HOST` — UDM controller IP
- `UNIFI_USER` / `UNIFI_PASS` — local Protect admin
- `NTFY_URL` / `NTFY_TOPIC` — push notification target
- `DATA_DIR` — SQLite + photo store
- `CAMERAS_CONFIG_PATH` — YAML with per-camera rules
- `FAMILY_MACS_PATH` — YAML mapping member name → phone MAC

## License

Apache-2.0
