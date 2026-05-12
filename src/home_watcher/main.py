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
from .bodies.db import BodyDB
from .bodies.reid import BodyReID
from .faces.db import FaceDB
from .faces.recognizer import FaceRecognizer, save_training_photo
from .faces.unknown_db import UnknownFaceDB
from .pets.db import PetDB
from .pets.detector import PetDetector
from .notifier.ntfy import NtfyNotifier
from .presence.unifi_clients import UnifiClientsLookup
from .protect.client import ProtectClient
from .protect.events import ProtectUpdate, event_camera_id, is_motion_event, smart_detect_types
from .protect.websocket import ProtectWebSocket

log = structlog.get_logger(__name__)


class AppState:
    settings: Settings
    cameras: dict[str, CameraConfig]
    face_db: FaceDB
    unknown_db: UnknownFaceDB
    recognizer: FaceRecognizer
    pet_db: PetDB
    pet_detector: PetDetector
    body_db: BodyDB
    body_reid: BodyReID
    protect: ProtectClient
    presence: UnifiClientsLookup
    notifier: NtfyNotifier
    ws_task: asyncio.Task[None] | None = None
    # Dedup state: camera_id -> last_notification_unixtime
    last_alerted: dict[str, float] = {}
    dedup_window_seconds: float = 120.0
    # Body crop dedup: camera_id -> last body-save unixtime
    last_body_saved: dict[str, float] = {}
    body_save_interval_seconds: float = 10.0
    min_body_crop_px: int = 40


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
    # Diagnostic: log every incoming update at debug level + a sampled count at info
    state._event_count = getattr(state, "_event_count", 0) + 1  # type: ignore[attr-defined]
    if state._event_count % 50 == 1:  # type: ignore[attr-defined]
        log.info(
            "ws_event_sample",
            count=state._event_count,  # type: ignore[attr-defined]
            model=update.model_key,
            action=update.action,
            data_keys=list(update.data.keys())[:8],
        )

    if not is_motion_event(update):
        return
    camera_id = event_camera_id(update)
    if not camera_id:
        log.warning("motion_event_without_camera_id",
                    model=update.model_key, action=update.action,
                    data_keys=list(update.data.keys()))
        return
    camera_name = state.protect.camera_name(camera_id)
    sd_types = smart_detect_types(update)
    log.info("motion_event_received", camera=camera_name, types=sd_types,
             data_keys=list(update.data.keys()))

    snapshot = await state.protect.fetch_snapshot(camera_id)
    if snapshot is None:
        log.warning("snapshot_unavailable", camera=camera_name)
        return

    # ALWAYS run face recognition on the snapshot — don't rely on Protect having
    # configured 'person' smart detection on this camera. If we find any face,
    # treat as a person event regardless of what Protect classified.
    faces = state.recognizer.recognize(snapshot)
    if faces and "person" not in sd_types:
        sd_types = [*sd_types, "person"]
        log.info("inferred_person_from_face", camera=camera_name,
                 face_count=len(faces))

    # YOLO person detection — even if face_rec already inferred person, we want
    # the bboxes so we can run Body Re-ID on each person region.
    body_matches: list[str] = []
    try:
        person_bboxes = state.pet_detector.detect_person_bboxes(snapshot)
    except Exception as exc:  # noqa: BLE001
        log.warning("yolo_person_detect_failed", error=str(exc))
        person_bboxes = []

    if person_bboxes and "person" not in sd_types:
        sd_types = [*sd_types, "person"]
        log.info("inferred_person_from_yolo", camera=camera_name,
                 person_count=len(person_bboxes))

    if not sd_types:
        log.info("motion_no_useful_classification", camera=camera_name)
        return

    log.info("motion_event", camera=camera_name, types=sd_types)
    if faces:
        _save_unknown_faces(faces, snapshot, camera_name)

    # Body Re-ID: crop each person bbox, extract embedding, match against known.
    # Save unknown bodies to queue for labeling.
    if person_bboxes:
        body_matches = _identify_bodies(person_bboxes, snapshot, camera_name, camera_id)
    if "animal" in sd_types:
        _detect_and_save_pets(snapshot, camera_name)
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
        body_matches=body_matches,
        body_person_count=len(person_bboxes),
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

    # Dedup: Protect emits two WS messages per motion (event/add + camera/update)
    # which both translate to the same user-visible alert. Skip if we already
    # alerted for this camera within dedup_window_seconds.
    import time as _time
    now_ts = _time.time()
    last = state.last_alerted.get(camera_id, 0.0)
    if now_ts - last < state.dedup_window_seconds:
        log.info("notification_deduped", camera=camera_name,
                 seconds_since_last=round(now_ts - last, 1))
        return
    state.last_alerted[camera_id] = now_ts

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


def _identify_bodies(
    person_bboxes: list[tuple[tuple[int, int, int, int], float]],
    snapshot: bytes,
    camera: str,
    camera_id: str,
) -> list[str]:
    """Run Body Re-ID on each detected person. Return list of matched subjects.
    Save unknown bodies to queue for labeling."""
    import time as _time
    from datetime import UTC, datetime

    matches: list[str] = []
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    body_dir = state.settings.data_dir / "unknown_bodies"
    body_dir.mkdir(parents=True, exist_ok=True)

    for idx, (bbox, conf) in enumerate(person_bboxes):
        top, right, bottom, left = bbox
        width = right - left
        height = bottom - top

        if width < state.min_body_crop_px or height < state.min_body_crop_px:
            log.debug("body_crop_too_small", camera=camera, w=width, h=height)
            continue

        try:
            crop_bytes = PetDetector.crop(snapshot, bbox, pad=10)
            embedding = state.body_reid.embed(crop_bytes)
        except Exception as exc:  # noqa: BLE001
            log.warning("body_reid_failed", error=str(exc))
            continue

        result = state.body_reid.match(embedding)

        if result.is_known:
            log.info(
                "body_matched",
                camera=camera,
                subject=result.matched_subject,
                similarity=round(result.similarity, 3),
            )
            assert result.matched_subject is not None
            matches.append(result.matched_subject)
            continue

        log.info(
            "body_unmatched",
            camera=camera,
            best_similarity=round(result.similarity, 3),
            crop_size=f"{width}x{height}",
        )

        now_ts = _time.time()
        last_saved = state.last_body_saved.get(camera_id, 0.0)
        if now_ts - last_saved < state.body_save_interval_seconds:
            continue
        state.last_body_saved[camera_id] = now_ts

        crop_filename = f"{timestamp}_{camera}_body_{idx}_crop.jpg"
        snap_filename = f"{timestamp}_{camera}_body_{idx}_snap.jpg"
        (body_dir / crop_filename).write_bytes(crop_bytes)
        (body_dir / snap_filename).write_bytes(snapshot)
        state.body_db.add_unknown(
            camera=camera,
            crop_filename=crop_filename,
            snapshot_filename=snap_filename,
            bbox=bbox,
            width_px=width,
            height_px=height,
            embedding=embedding,
        )
        log.info(
            "body_unknown_saved",
            camera=camera,
            best_similarity=round(result.similarity, 3),
        )
    return matches


def _detect_and_save_pets(snapshot: bytes, camera: str) -> None:
    """Run YOLO on snapshot, save any animal detections to the unknown_pets queue."""
    from datetime import UTC, datetime

    try:
        pets = state.pet_detector.detect(snapshot)
    except Exception as exc:  # noqa: BLE001
        log.warning("pet_detect_failed", error=str(exc), camera=camera)
        return
    if not pets:
        log.info("pet_event_no_yolo_match", camera=camera)
        return

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    pet_dir = state.settings.data_dir / "unknown_pets"
    pet_dir.mkdir(parents=True, exist_ok=True)

    for idx, pet in enumerate(pets):
        crop_bytes = PetDetector.crop(snapshot, pet.bbox)
        crop_filename = f"{timestamp}_{camera}_{pet.species}_{idx}_crop.jpg"
        snap_filename = f"{timestamp}_{camera}_{pet.species}_{idx}_snap.jpg"
        (pet_dir / crop_filename).write_bytes(crop_bytes)
        (pet_dir / snap_filename).write_bytes(snapshot)
        state.pet_db.add_unknown(
            camera=camera,
            species=pet.species,
            confidence=pet.confidence,
            crop_filename=crop_filename,
            snapshot_filename=snap_filename,
            bbox=pet.bbox,
            width_px=pet.width_px,
            height_px=pet.height_px,
        )
        log.info("pet_detected", camera=camera, species=pet.species,
                 confidence=round(pet.confidence, 2))


def _save_unknown_faces(faces: list, snapshot: bytes, camera: str) -> None:
    """Save crop + metadata for any face that's unknown or too uncertain."""
    from datetime import UTC, datetime

    from .faces.recognizer import FaceRecognizer

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    snapshot_dir = state.settings.data_dir / "unknown"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for idx, face in enumerate(faces):
        if face.is_known:
            continue
        if face.width_px < state.settings.min_face_width_px:
            continue
        if face.embedding is None:
            continue

        crop_bytes = FaceRecognizer.crop_face(snapshot, face.bbox)
        crop_filename = f"{timestamp}_{camera}_{idx}_crop.jpg"
        snap_filename = f"{timestamp}_{camera}_{idx}_snap.jpg"
        (snapshot_dir / crop_filename).write_bytes(crop_bytes)
        (snapshot_dir / snap_filename).write_bytes(snapshot)
        state.unknown_db.add(
            camera=camera,
            crop_filename=crop_filename,
            snapshot_filename=snap_filename,
            bbox=face.bbox,
            width_px=face.width_px,
            embedding=face.embedding,
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
    state.unknown_db = UnknownFaceDB(settings.data_dir / "unknown.db")
    state.recognizer = FaceRecognizer(
        state.face_db,
        tolerance=settings.face_tolerance,
        min_face_width_px=settings.min_face_width_px,
    )
    state.pet_db = PetDB(settings.data_dir / "pets.db")
    state.pet_detector = PetDetector()
    state.body_db = BodyDB(settings.data_dir / "bodies.db")
    state.body_reid = BodyReID(similarity_threshold=settings.body_similarity_threshold)
    state.body_reid.reload(state.body_db.all_known_by_subject())
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


@app.get("/api/unknown")
async def list_unknown(
    s: Annotated[AppState, Depends(_get_state)], limit: int = 100
) -> list[dict[str, object]]:
    rows = s.unknown_db.list_unlabeled(limit=limit)
    return [
        {
            "id": r.id,
            "detected_at": r.detected_at.isoformat(),
            "camera": r.camera,
            "width_px": r.width_px,
            "crop_url": f"/api/unknown/{r.id}/crop",
            "snapshot_url": f"/api/unknown/{r.id}/snapshot",
        }
        for r in rows
    ]


@app.get("/api/unknown/{face_id}/crop")
async def get_unknown_crop(
    face_id: int, s: Annotated[AppState, Depends(_get_state)]
):
    from fastapi.responses import FileResponse

    row = s.unknown_db.get(face_id)
    if not row:
        raise HTTPException(status_code=404)
    return FileResponse(s.settings.data_dir / "unknown" / row.crop_filename)


@app.get("/api/unknown/{face_id}/snapshot")
async def get_unknown_snapshot(
    face_id: int, s: Annotated[AppState, Depends(_get_state)]
):
    from fastapi.responses import FileResponse

    row = s.unknown_db.get(face_id)
    if not row:
        raise HTTPException(status_code=404)
    return FileResponse(s.settings.data_dir / "unknown" / row.snapshot_filename)


@app.post("/api/unknown/{face_id}/label")
async def label_unknown(
    face_id: int,
    body: dict[str, str],
    s: Annotated[AppState, Depends(_get_state)],
) -> dict[str, str | int]:
    subject = body.get("subject", "").strip()
    if not subject:
        raise HTTPException(status_code=400, detail="subject is required")
    row = s.unknown_db.get(face_id)
    if not row:
        raise HTTPException(status_code=404)
    crop_path = s.settings.data_dir / "unknown" / row.crop_filename
    crop_bytes = crop_path.read_bytes() if crop_path.exists() else b""
    if crop_bytes:
        save_training_photo(s.settings.data_dir, subject, row.crop_filename, crop_bytes)
    db_id = s.face_db.add(subject, row.crop_filename, row.embedding)
    s.unknown_db.mark_labeled(face_id)
    s.recognizer.reload()
    return {"face_db_id": db_id, "subject": subject}


@app.delete("/api/unknown/{face_id}")
async def discard_unknown(
    face_id: int, s: Annotated[AppState, Depends(_get_state)]
) -> dict[str, str]:
    if s.unknown_db.get(face_id) is None:
        raise HTTPException(status_code=404)
    s.unknown_db.mark_discarded(face_id)
    return {"status": "discarded"}


@app.get("/api/pets/subjects")
async def list_pet_subjects(s: Annotated[AppState, Depends(_get_state)]) -> dict[str, int]:
    return s.pet_db.list_known_subjects()


@app.get("/api/pets/unknown")
async def list_unknown_pets(
    s: Annotated[AppState, Depends(_get_state)], limit: int = 100
) -> list[dict[str, object]]:
    rows = s.pet_db.list_unlabeled(limit=limit)
    return [
        {
            "id": r.id,
            "detected_at": r.detected_at.isoformat(),
            "camera": r.camera,
            "species": r.species,
            "confidence": round(r.confidence, 2),
            "width_px": r.width_px,
            "height_px": r.height_px,
            "crop_url": f"/api/pets/unknown/{r.id}/crop",
            "snapshot_url": f"/api/pets/unknown/{r.id}/snapshot",
        }
        for r in rows
    ]


@app.get("/api/pets/unknown/{pet_id}/crop")
async def get_unknown_pet_crop(
    pet_id: int, s: Annotated[AppState, Depends(_get_state)]
):
    from fastapi.responses import FileResponse

    row = s.pet_db.get_unknown(pet_id)
    if not row:
        raise HTTPException(status_code=404)
    return FileResponse(s.settings.data_dir / "unknown_pets" / row.crop_filename)


@app.get("/api/pets/unknown/{pet_id}/snapshot")
async def get_unknown_pet_snapshot(
    pet_id: int, s: Annotated[AppState, Depends(_get_state)]
):
    from fastapi.responses import FileResponse

    row = s.pet_db.get_unknown(pet_id)
    if not row:
        raise HTTPException(status_code=404)
    return FileResponse(s.settings.data_dir / "unknown_pets" / row.snapshot_filename)


@app.post("/api/pets/unknown/{pet_id}/label")
async def label_unknown_pet(
    pet_id: int,
    body: dict[str, str],
    s: Annotated[AppState, Depends(_get_state)],
) -> dict[str, str | int]:
    subject = body.get("subject", "").strip()
    if not subject:
        raise HTTPException(status_code=400, detail="subject is required")
    row = s.pet_db.get_unknown(pet_id)
    if not row:
        raise HTTPException(status_code=404)
    # Move/copy the crop into the known pets dir
    crop_src = s.settings.data_dir / "unknown_pets" / row.crop_filename
    known_dir = s.settings.data_dir / "pets" / subject
    known_dir.mkdir(parents=True, exist_ok=True)
    if crop_src.exists():
        (known_dir / row.crop_filename).write_bytes(crop_src.read_bytes())
    db_id = s.pet_db.add_known(subject, row.species, row.crop_filename)
    s.pet_db.mark_labeled(pet_id)
    return {"known_id": db_id, "subject": subject, "species": row.species}


@app.delete("/api/pets/unknown/{pet_id}")
async def discard_unknown_pet(
    pet_id: int, s: Annotated[AppState, Depends(_get_state)]
) -> dict[str, str]:
    if s.pet_db.get_unknown(pet_id) is None:
        raise HTTPException(status_code=404)
    s.pet_db.mark_discarded(pet_id)
    return {"status": "discarded"}


@app.get("/api/bodies/subjects")
async def list_body_subjects(s: Annotated[AppState, Depends(_get_state)]) -> dict[str, int]:
    return s.body_db.list_known_subjects()


@app.get("/api/bodies/unknown")
async def list_unknown_bodies(
    s: Annotated[AppState, Depends(_get_state)], limit: int = 100
) -> list[dict[str, object]]:
    rows = s.body_db.list_unlabeled(limit=limit)
    return [
        {
            "id": r.id,
            "detected_at": r.detected_at.isoformat(),
            "camera": r.camera,
            "width_px": r.width_px,
            "height_px": r.height_px,
            "crop_url": f"/api/bodies/unknown/{r.id}/crop",
            "snapshot_url": f"/api/bodies/unknown/{r.id}/snapshot",
        }
        for r in rows
    ]


@app.get("/api/bodies/unknown/{body_id}/crop")
async def get_unknown_body_crop(
    body_id: int, s: Annotated[AppState, Depends(_get_state)]
):
    from fastapi.responses import FileResponse

    row = s.body_db.get_unknown(body_id)
    if not row:
        raise HTTPException(status_code=404)
    return FileResponse(s.settings.data_dir / "unknown_bodies" / row.crop_filename)


@app.get("/api/bodies/unknown/{body_id}/snapshot")
async def get_unknown_body_snapshot(
    body_id: int, s: Annotated[AppState, Depends(_get_state)]
):
    from fastapi.responses import FileResponse

    row = s.body_db.get_unknown(body_id)
    if not row:
        raise HTTPException(status_code=404)
    return FileResponse(s.settings.data_dir / "unknown_bodies" / row.snapshot_filename)


@app.post("/api/bodies/unknown/{body_id}/label")
async def label_unknown_body(
    body_id: int,
    body: dict[str, str],
    s: Annotated[AppState, Depends(_get_state)],
) -> dict[str, str | int]:
    subject = body.get("subject", "").strip()
    if not subject:
        raise HTTPException(status_code=400, detail="subject is required")
    row = s.body_db.get_unknown(body_id)
    if not row:
        raise HTTPException(status_code=404)
    db_id = s.body_db.add_known(subject, row.crop_filename, row.embedding)
    s.body_db.mark_labeled(body_id)
    s.body_reid.reload(s.body_db.all_known_by_subject())
    return {"known_id": db_id, "subject": subject}


@app.delete("/api/bodies/unknown/{body_id}")
async def discard_unknown_body(
    body_id: int, s: Annotated[AppState, Depends(_get_state)]
) -> dict[str, str]:
    if s.body_db.get_unknown(body_id) is None:
        raise HTTPException(status_code=404)
    s.body_db.mark_discarded(body_id)
    return {"status": "discarded"}


@app.post("/api/backfill")
async def backfill_old_events(
    s: Annotated[AppState, Depends(_get_state)],
    days: int = 7,
    max_events: int = 500,
) -> dict[str, int]:
    """Pull historical events from Protect for the last `days` days,
    run face detection on each event thumbnail, queue unknown faces
    for labeling in the UI.

    Useful right after setup to build training data from existing footage.
    """
    import time as _time
    from datetime import UTC, datetime

    now_ms = int(_time.time() * 1000)
    start_ms = now_ms - days * 24 * 3600 * 1000

    events = await s.protect.list_events(
        start_ms, now_ms,
        types=["motion", "smartDetectZone"],
        limit=max_events,
    )
    log.info("backfill_started", days=days, event_count=len(events))

    faces_found = 0
    events_with_thumbnail = 0
    events_with_no_face = 0
    events_with_known_face = 0

    snapshot_dir = s.settings.data_dir / "unknown"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for ev in events:
        event_id = ev.get("id")
        camera_id = ev.get("camera")
        if not event_id or not camera_id:
            continue
        camera_name = s.protect.camera_name(camera_id)
        thumb = await s.protect.fetch_event_thumbnail(event_id)
        if thumb is None:
            continue
        events_with_thumbnail += 1

        try:
            # Thumbnails are 360x360 — upsample=2 lets dlib find faces down to ~40px.
            detected = s.recognizer.recognize(thumb, upsample=2)
        except Exception as exc:  # noqa: BLE001
            log.warning("backfill_face_recog_failed", event_id=event_id, error=str(exc))
            continue

        if not detected:
            events_with_no_face += 1
            continue

        for idx, face in enumerate(detected):
            if face.is_known:
                events_with_known_face += 1
                continue
            # Lower threshold for backfill: 360x360 thumbnails mean even small
            # faces are worth keeping for training data.
            if face.width_px < 30:
                continue
            if face.embedding is None:
                continue

            crop_bytes = type(s.recognizer).crop_face(thumb, face.bbox)
            ts = datetime.fromtimestamp(ev.get("start", now_ms) / 1000, UTC).strftime("%Y%m%dT%H%M%S")
            crop_filename = f"backfill_{ts}_{camera_name}_{event_id[:8]}_{idx}_crop.jpg"
            snap_filename = f"backfill_{ts}_{camera_name}_{event_id[:8]}_{idx}_snap.jpg"
            (snapshot_dir / crop_filename).write_bytes(crop_bytes)
            (snapshot_dir / snap_filename).write_bytes(thumb)
            s.unknown_db.add(
                camera=camera_name,
                crop_filename=crop_filename,
                snapshot_filename=snap_filename,
                bbox=face.bbox,
                width_px=face.width_px,
                embedding=face.embedding,
            )
            faces_found += 1

    log.info(
        "backfill_done",
        events_total=len(events),
        events_with_thumbnail=events_with_thumbnail,
        events_no_face=events_with_no_face,
        events_known_face=events_with_known_face,
        new_unknown_faces=faces_found,
    )
    return {
        "events_total": len(events),
        "events_with_thumbnail": events_with_thumbnail,
        "events_no_face": events_with_no_face,
        "events_known_face": events_with_known_face,
        "new_unknown_faces": faces_found,
    }


@app.get("/", response_class=None)
async def root():
    from fastapi.responses import HTMLResponse
    from pathlib import Path as P

    html_path = P(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse(_INLINE_HTML)


_INLINE_HTML = """<!doctype html>
<html lang="sv">
<head>
<meta charset="utf-8">
<title>home-watcher — labeling</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
         margin: 0; padding: 1rem; background: #111; color: #eee; }
  h1, h2 { margin-top: 0; }
  h2 { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #333; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 1rem; }
  .card { background: #1c1c1c; border: 1px solid #333; border-radius: 8px; padding: 0.75rem; }
  .card img { width: 100%; height: auto; border-radius: 4px; display: block; }
  .meta { font-size: 0.85rem; color: #999; margin: 0.5rem 0; }
  .row { display: flex; gap: 0.5rem; margin-top: 0.5rem; }
  input[type=text] { flex: 1; padding: 0.4rem 0.6rem; background: #2a2a2a; color: #eee;
                     border: 1px solid #444; border-radius: 4px; }
  button { padding: 0.4rem 0.8rem; border: none; border-radius: 4px;
           background: #2563eb; color: white; cursor: pointer; }
  button.secondary { background: #444; }
  button:hover { opacity: 0.9; }
  details { margin-top: 0.5rem; font-size: 0.85rem; }
  details summary { cursor: pointer; color: #888; }
  .known { font-size: 0.85rem; color: #888; margin-bottom: 1rem; }
  .tag { display: inline-block; background: #333; color: #ccc; padding: 0.1rem 0.4rem;
         border-radius: 3px; font-size: 0.75rem; margin-right: 0.25rem; }
</style>
</head>
<body>

<h1>Okända personer (kropp)</h1>
<div class="known" id="known-bodies"></div>
<div class="grid" id="grid-bodies"></div>

<h2>Okända ansikten</h2>
<div class="known" id="known-faces"></div>
<div class="grid" id="grid-faces"></div>

<h2>Okända djur</h2>
<div class="known" id="known-pets"></div>
<div class="grid" id="grid-pets"></div>

<script>
  async function loadKnown(url, target, prefix) {
    const r = await fetch(url);
    const data = await r.json();
    const parts = Object.entries(data).map(([n, c]) => `${n} (${c})`);
    target.textContent = parts.length ? prefix + ': ' + parts.join(', ') : 'Inga tränade än';
  }

  async function loadFaces() {
    const r = await fetch('/api/unknown');
    const items = await r.json();
    const grid = document.getElementById('grid-faces');
    grid.innerHTML = items.length === 0 ? '<p>Inga okända ansikten i kön.</p>' : '';
    for (const f of items) {
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        <img src="${f.crop_url}" alt="face">
        <div class="meta">${f.camera} — ${new Date(f.detected_at).toLocaleString('sv-SE')} — ${f.width_px}px</div>
        <div class="row">
          <input type="text" placeholder="Namn (t.ex. Malin)" id="fn-${f.id}">
          <button onclick="labelFace(${f.id})">Spara</button>
        </div>
        <div class="row"><button class="secondary" onclick="discardFace(${f.id})">Skippa</button></div>
        <details><summary>Full bild</summary><img src="${f.snapshot_url}" style="margin-top:0.5rem"></details>`;
      grid.appendChild(card);
    }
  }

  async function loadPets() {
    const r = await fetch('/api/pets/unknown');
    const items = await r.json();
    const grid = document.getElementById('grid-pets');
    grid.innerHTML = items.length === 0 ? '<p>Inga okända djur i kön.</p>' : '';
    for (const p of items) {
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        <img src="${p.crop_url}" alt="pet">
        <div class="meta">
          <span class="tag">${p.species}</span>
          <span class="tag">${Math.round(p.confidence*100)}%</span>
          ${p.camera} — ${new Date(p.detected_at).toLocaleString('sv-SE')}
        </div>
        <div class="row">
          <input type="text" placeholder="Namn (t.ex. Bella)" id="pn-${p.id}">
          <button onclick="labelPet(${p.id})">Spara</button>
        </div>
        <div class="row"><button class="secondary" onclick="discardPet(${p.id})">Skippa</button></div>
        <details><summary>Full bild</summary><img src="${p.snapshot_url}" style="margin-top:0.5rem"></details>`;
      grid.appendChild(card);
    }
  }

  async function postLabel(url, id, name) {
    return fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({subject: name}),
    });
  }

  async function labelFace(id) {
    const name = document.getElementById('fn-' + id).value.trim();
    if (!name) { alert('Ange namn'); return; }
    const r = await postLabel(`/api/unknown/${id}/label`, id, name);
    if (r.ok) await refresh(); else alert('Fel: ' + r.status);
  }
  async function labelPet(id) {
    const name = document.getElementById('pn-' + id).value.trim();
    if (!name) { alert('Ange namn'); return; }
    const r = await postLabel(`/api/pets/unknown/${id}/label`, id, name);
    if (r.ok) await refresh(); else alert('Fel: ' + r.status);
  }
  async function discardFace(id) {
    await fetch(`/api/unknown/${id}`, {method: 'DELETE'});
    await loadFaces();
  }
  async function discardPet(id) {
    await fetch(`/api/pets/unknown/${id}`, {method: 'DELETE'});
    await loadPets();
  }

  async function loadBodies() {
    const r = await fetch('/api/bodies/unknown');
    const items = await r.json();
    const grid = document.getElementById('grid-bodies');
    grid.innerHTML = items.length === 0 ? '<p>Inga okända personer i kön.</p>' : '';
    for (const b of items) {
      const card = document.createElement('div');
      card.className = 'card';
      card.innerHTML = `
        <img src="${b.crop_url}" alt="body">
        <div class="meta">${b.camera} — ${new Date(b.detected_at).toLocaleString('sv-SE')} — ${b.width_px}×${b.height_px}px</div>
        <div class="row">
          <input type="text" placeholder="Namn (t.ex. Malin)" id="bn-${b.id}">
          <button onclick="labelBody(${b.id})">Spara</button>
        </div>
        <div class="row"><button class="secondary" onclick="discardBody(${b.id})">Skippa</button></div>
        <details><summary>Full bild</summary><img src="${b.snapshot_url}" style="margin-top:0.5rem"></details>`;
      grid.appendChild(card);
    }
  }
  async function labelBody(id) {
    const name = document.getElementById('bn-' + id).value.trim();
    if (!name) { alert('Ange namn'); return; }
    const r = await postLabel(`/api/bodies/unknown/${id}/label`, id, name);
    if (r.ok) await refresh(); else alert('Fel: ' + r.status);
  }
  async function discardBody(id) {
    await fetch(`/api/bodies/unknown/${id}`, {method: 'DELETE'});
    await loadBodies();
  }

  async function refresh() {
    await Promise.all([
      loadKnown('/api/subjects', document.getElementById('known-faces'), 'Tränade ansikten'),
      loadKnown('/api/pets/subjects', document.getElementById('known-pets'), 'Tränade djur'),
      loadKnown('/api/bodies/subjects', document.getElementById('known-bodies'), 'Tränade personer'),
      loadFaces(),
      loadPets(),
      loadBodies(),
    ]);
  }
  refresh();
  setInterval(refresh, 30000);
</script>
</body>
</html>"""


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
