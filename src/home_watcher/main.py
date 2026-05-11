"""home-watcher entry point.

Starts:
  - Background asyncio task: WebSocket subscriber → event processor pipeline
  - FastAPI app: admin endpoints (train faces, list subjects, health)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

import structlog
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from .config import CameraConfig, Settings, load_cameras, load_family_macs
from .decision.scorer import (
    Decision,
    DecisionResult,
    ScoringContext,
    decide,
)
from .faces.db import FaceDB
from .faces.recognizer import FaceRecognizer, save_training_photo
from .notifier.ntfy import NtfyNotifier
from .presence.unifi_clients import UnifiClientsLookup
from .protect.client import ProtectClient
from .protect.events import ProtectUpdate, is_motion_event, smart_detect_types
from .protect.websocket import ProtectWebSocket

log = structlog.get_logger(__name__)


class AppState:
    settings: Settings
    cameras: dict[str, CameraConfig]
    face_db: FaceDB
    recognizer: FaceRecognizer
    protect: ProtectClient
    presence: UnifiClientsLookup
    notifier: NtfyNotifier
    ws_task: asyncio.Task[None] | None = None


state = AppState()


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )


async def _handle_event(update: ProtectUpdate) -> None:
    if not is_motion_event(update):
        return
    camera_id = update.id
    camera_name = state.protect.camera_name(camera_id)
    sd_types = smart_detect_types(update)
    if not sd_types:
        return  # Pure motion without classification — ignore for now

    log.info("motion_event", camera=camera_name, types=sd_types)

    snapshot = await state.protect.fetch_snapshot(camera_id)
    if snapshot is None:
        log.warning("snapshot_unavailable", camera=camera_name)
        return

    faces = state.recognizer.recognize(snapshot) if "person" in sd_types else []
    family_home = await state.presence.family_at_home()
    camera_cfg = state.cameras.get(camera_name, CameraConfig())

    ctx = ScoringContext(
        camera_id=camera_id,
        camera_name=camera_name,
        smart_detect_types=sd_types,
        faces=faces,
        now=datetime.now(),
        family_at_home=family_home,
        camera_cfg=camera_cfg,
    )
    result = decide(
        ctx,
        alert_threshold=state.settings.alert_score_threshold,
        min_face_width_px=state.settings.min_face_width_px,
    )
    log.info(
        "decision",
        camera=camera_name,
        decision=result.decision.value,
        score=round(result.score, 2),
        reasons=result.reasons,
        matched=result.matched_subjects,
        family_home=family_home,
        face_count=len(faces),
    )

    await _send_notification(result, camera_id, camera_name, sd_types, snapshot)


async def _send_notification(
    result: DecisionResult,
    camera_id: str,
    camera_name: str,
    sd_types: list[str],
    snapshot: bytes,
) -> None:
    click_url = f"https://{state.settings.unifi_host}/protect/cameras/{camera_id}"

    if result.decision == Decision.ALERT:
        await state.notifier.send(
            title=f"Okänd rörelse vid {camera_name}",
            message=f"Score: {result.score:.2f}\n" + "\n".join(result.reasons),
            priority=4,
            tags=["warning", "house"],
            image_bytes=snapshot,
            click_url=click_url,
        )
    elif result.decision == Decision.NOTIFY_ANIMAL:
        await state.notifier.send(
            title=f"Djur vid {camera_name}",
            message="Smart detection: " + ", ".join(sd_types),
            priority=2,
            tags=["paw_prints"],
            image_bytes=snapshot,
            click_url=click_url,
        )


async def _ws_loop() -> None:
    ws = ProtectWebSocket(state.protect)
    try:
        async for update in ws.stream():
            try:
                await _handle_event(update)
            except Exception as exc:  # noqa: BLE001
                log.error("event_handler_error", error=str(exc), update_id=update.id)
    except asyncio.CancelledError:
        ws.stop()
        raise


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()  # type: ignore[call-arg]
    _setup_logging(settings.log_level)
    state.settings = settings

    state.cameras = load_cameras(settings.cameras_config_path)
    log.info("cameras_loaded", count=len(state.cameras))

    family_macs = load_family_macs(settings.family_macs_path)
    log.info("family_macs_loaded", count=len(family_macs))

    state.face_db = FaceDB(settings.data_dir / "faces.db")
    state.recognizer = FaceRecognizer(
        state.face_db,
        tolerance=settings.face_tolerance,
        min_face_width_px=settings.min_face_width_px,
    )
    state.protect = ProtectClient(
        host=settings.unifi_host,
        username=settings.unifi_user,
        password=settings.unifi_pass,
        verify_tls=settings.unifi_verify_tls,
    )
    state.presence = UnifiClientsLookup(
        base_url=f"https://{settings.unifi_host}",
        username=settings.unifi_user,
        password=settings.unifi_pass,
        family_macs=family_macs,
        verify_tls=settings.unifi_verify_tls,
    )
    state.notifier = NtfyNotifier(
        base_url=settings.ntfy_url,
        topic=settings.ntfy_topic,
        token=settings.ntfy_token,
    )

    state.ws_task = asyncio.create_task(_ws_loop())
    yield

    if state.ws_task:
        state.ws_task.cancel()
        try:
            await state.ws_task
        except asyncio.CancelledError:
            pass
    await state.protect.close()
    await state.presence.close()


app = FastAPI(title="home-watcher", lifespan=lifespan)


def _get_state() -> AppState:
    return state


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/subjects")
async def list_subjects(s: Annotated[AppState, Depends(_get_state)]) -> dict[str, int]:
    return s.face_db.list_subjects()


@app.post("/api/subjects/{name}/photos")
async def add_subject_photo(
    name: str,
    photo: UploadFile = File(...),  # noqa: B008
    s: Annotated[AppState, Depends(_get_state)] = ...,
) -> dict[str, str | int]:
    image_bytes = await photo.read()
    encoding = s.recognizer.encode_for_training(image_bytes)
    if encoding is None:
        raise HTTPException(status_code=400, detail="No face detected in photo")
    filename = photo.filename or "upload.jpg"
    save_training_photo(s.settings.data_dir, name, filename, image_bytes)
    row_id = s.face_db.add(name, filename, encoding)
    s.recognizer.reload()
    return {"id": row_id, "subject": name, "filename": filename}


@app.delete("/api/subjects/{name}")
async def delete_subject(
    name: str, s: Annotated[AppState, Depends(_get_state)]
) -> dict[str, int]:
    n = s.face_db.delete_subject(name)
    s.recognizer.reload()
    return {"deleted": n}


@app.post("/api/reload")
async def reload_faces(s: Annotated[AppState, Depends(_get_state)]) -> dict[str, list[str]]:
    s.recognizer.reload()
    return {"subjects": s.recognizer.known_subjects()}


@app.post("/api/test-recognize")
async def test_recognize(
    photo: UploadFile = File(...),  # noqa: B008
    s: Annotated[AppState, Depends(_get_state)] = ...,
) -> list[dict[str, object]]:
    image_bytes = await photo.read()
    faces = s.recognizer.recognize(image_bytes)
    return [
        {
            "matched_subject": f.matched_subject,
            "distance": round(f.distance, 3),
            "width_px": f.width_px,
            "bbox": list(f.bbox),
        }
        for f in faces
    ]


def main() -> None:
    settings = Settings()  # type: ignore[call-arg]
    uvicorn.run(
        "home_watcher.main:app",
        host=settings.bind_host,
        port=settings.bind_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
