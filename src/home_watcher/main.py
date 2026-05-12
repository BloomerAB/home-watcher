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
from .skeleton.analyzer import SkeletonAnalyzer
from .skeleton.db import SkeletonDB
from .skeleton.matcher import SkeletonMatcher
from .trajectory.db import TrajectoryDB
from .trajectory.matcher import TrajectoryMatcher
from .trajectory.tracker import BurstTracker, Trajectory
from .faces.db import FaceDB
from .faces.recognizer import FaceRecognizer, save_training_photo
from .faces.unknown_db import UnknownFaceDB
from .pets.db import PetDB
from .pets.detector import PetDetector
from .pets.reid import PetReID
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
    pet_reid: PetReID
    body_db: BodyDB
    body_reid: BodyReID
    trajectory_db: TrajectoryDB
    trajectory_matcher: TrajectoryMatcher
    skeleton_analyzer: SkeletonAnalyzer
    skeleton_db: SkeletonDB
    skeleton_matcher: SkeletonMatcher
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
    analyzed_event_ids: set[str] = set()


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
    if person_bboxes:
        body_matches = _identify_bodies(person_bboxes, snapshot, camera_name, camera_id)
    matched_pets: list[str] = []
    if "animal" in sd_types:
        matched_pets = _detect_and_save_pets(snapshot, camera_name)

    # Trajectory tracking: burst-capture 2 more snapshots and track positions.
    import time as _time
    trajectory_matches: list[str] = []
    unmatched_trajectories: list[Trajectory] = []
    skeleton_matches: list[str] = []
    if person_bboxes and "person" in sd_types:
        trajectory_matches, unmatched_trajectories, skeleton_matches = (
            await _track_and_analyze(
                camera_id, camera_name, person_bboxes, snapshot, _time.time(),
            )
        )
    family_members = await state.presence.family_members_at_home()
    family_home = len(family_members) > 0
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
        family_members_home=family_members,
        trajectory_matches=trajectory_matches,
        skeleton_matches=skeleton_matches,
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

    # Auto-label: when decision is KNOWN_FAMILY via presence-count, save
    # unmatched trajectories and skeleton profiles as known — training data
    # builds up automatically without manual labeling.
    if result.decision == Decision.KNOWN_FAMILY and family_members:
        subject = family_members[0]
        for traj in unmatched_trajectories:
            state.trajectory_db.add(traj, subject=subject)
        state.trajectory_matcher.reload(state.trajectory_db.known_by_camera())
        log.info(
            "auto_labeled",
            camera=camera_name,
            subject=subject,
            trajectories=len(unmatched_trajectories),
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

    event_id = update.id if update.model_key == "event" else None
    await _send_notification(result, camera_id, camera_name, sd_types, snapshot, event_id, matched_pets)


async def _send_notification(
    result: DecisionResult,
    camera_id: str,
    camera_name: str,
    sd_types: list[str],
    snapshot: bytes,
    event_id: str | None = None,
    matched_pets: list[str] | None = None,
) -> None:
    if event_id:
        click_url = f"https://{state.settings.unifi_host}/protect/events/{event_id}"
    else:
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
        pets = matched_pets or []
        if pets:
            title = f"{', '.join(pets)} vid {camera_name}"
            msg = f"Känt husdjur: {', '.join(pets)}"
        else:
            title = f"Okänt djur vid {camera_name}"
            msg = "Smart detection: " + ", ".join(sd_types)
        await state.notifier.send(
            title=title,
            message=msg,
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


async def _track_and_analyze(
    camera_id: str,
    camera_name: str,
    initial_bboxes: list[tuple[tuple[int, int, int, int], float]],
    initial_snapshot: bytes,
    t0: float,
) -> tuple[list[str], list[Trajectory], list[str]]:
    """Burst-capture snapshots for trajectory tracking + skeleton analysis.

    Returns (trajectory_matches, unmatched_trajectories, skeleton_matches).
    """
    tracker = BurstTracker(camera=camera_name)
    tracker.add_frame(initial_bboxes, t0)
    burst_snapshots = [initial_snapshot]

    for i in range(1, 3):
        await asyncio.sleep(2.0)
        snap = await state.protect.fetch_snapshot(camera_id)
        if snap is None:
            continue
        burst_snapshots.append(snap)
        try:
            bboxes = state.pet_detector.detect_person_bboxes(snap)
        except Exception:  # noqa: BLE001
            continue
        tracker.add_frame(bboxes, t0 + i * 2.0)

    # --- Trajectory matching ---
    trajectories = tracker.get_trajectories()
    traj_matches: list[str] = []
    unmatched: list[Trajectory] = []
    for traj in trajectories:
        if traj.is_stationary:
            continue
        result = state.trajectory_matcher.match(traj)
        if result.is_known:
            log.info(
                "trajectory_matched",
                camera=camera_name,
                subject=result.matched_subject,
                similarity=round(result.similarity, 3),
            )
            assert result.matched_subject is not None
            traj_matches.append(result.matched_subject)
        else:
            unmatched.append(traj)

    # --- Skeleton analysis (proportions + gait) ---
    skel_matches: list[str] = []
    try:
        profile = state.skeleton_analyzer.build_profile(burst_snapshots)
        if profile is not None:
            vec = profile.to_vector()
            skel_result = state.skeleton_matcher.match(vec)
            if skel_result.is_known:
                log.info(
                    "skeleton_matched",
                    camera=camera_name,
                    subject=skel_result.matched_subject,
                    similarity=round(skel_result.similarity, 3),
                )
                assert skel_result.matched_subject is not None
                skel_matches.append(skel_result.matched_subject)
            else:
                props = profile.proportions
                state.skeleton_db.add(
                    camera=camera_name,
                    profile_vector=vec,
                    shoulder_ratio=props.shoulder_ratio if props else 0.0,
                    torso_ratio=props.torso_ratio if props else 0.0,
                    leg_ratio=props.leg_ratio if props else 0.0,
                    arm_ratio=props.arm_ratio if props else 0.0,
                    height_px=props.height_px if props else 0.0,
                )
                log.info(
                    "skeleton_unknown_saved",
                    camera=camera_name,
                    best_similarity=round(skel_result.similarity, 3),
                    proportions={
                        "shoulder": round(props.shoulder_ratio, 3) if props else 0,
                        "torso": round(props.torso_ratio, 3) if props else 0,
                        "leg": round(props.leg_ratio, 3) if props else 0,
                        "arm": round(props.arm_ratio, 3) if props else 0,
                        "height_px": round(props.height_px, 1) if props else 0,
                    },
                )
    except Exception as exc:  # noqa: BLE001
        log.warning("skeleton_analysis_failed", error=str(exc))

    return traj_matches, unmatched, skel_matches


def _detect_and_save_pets(snapshot: bytes, camera: str) -> list[str]:
    """Run YOLO on snapshot, match against known pets, save unknowns for labeling.

    Returns list of matched pet names.
    """
    from datetime import UTC, datetime

    try:
        pets = state.pet_detector.detect(snapshot)
    except Exception as exc:  # noqa: BLE001
        log.warning("pet_detect_failed", error=str(exc), camera=camera)
        return []
    if not pets:
        log.info("pet_event_no_yolo_match", camera=camera)
        return []

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    pet_dir = state.settings.data_dir / "unknown_pets"
    pet_dir.mkdir(parents=True, exist_ok=True)
    matched_names: list[str] = []

    for idx, pet in enumerate(pets):
        crop_bytes = PetDetector.crop(snapshot, pet.bbox)
        embedding = state.pet_reid.embed(crop_bytes)
        match = state.pet_reid.match(embedding)

        if match.is_known:
            log.info(
                "pet_matched",
                camera=camera,
                subject=match.matched_subject,
                species=pet.species,
                similarity=round(match.similarity, 3),
            )
            matched_names.append(match.matched_subject or pet.species)
            continue

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
        log.info(
            "pet_unknown",
            camera=camera,
            species=pet.species,
            confidence=round(pet.confidence, 2),
            best_similarity=round(match.similarity, 3),
        )

    return matched_names


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
    state.pet_reid = PetReID()
    pet_known = state.pet_db.all_known_by_subject()
    state.pet_reid.reload(pet_known, state.pet_db.species_by_subject())
    state.body_db = BodyDB(settings.data_dir / "bodies.db")
    state.body_reid = BodyReID(similarity_threshold=settings.body_similarity_threshold)
    body_known = state.body_db.all_known_by_subject()
    state.body_reid.reload(body_known)
    state.trajectory_db = TrajectoryDB(settings.data_dir / "trajectories.db")
    state.trajectory_matcher = TrajectoryMatcher()
    traj_known = state.trajectory_db.known_by_camera()
    state.trajectory_matcher.reload(traj_known)
    state.skeleton_analyzer = SkeletonAnalyzer()
    state.skeleton_db = SkeletonDB(settings.data_dir / "skeletons.db")
    state.skeleton_matcher = SkeletonMatcher()
    skel_known = state.skeleton_db.known_by_subject()
    state.skeleton_matcher.reload(skel_known)
    log.info(
        "models_loaded",
        body_subjects=list(body_known.keys()),
        body_embeddings=sum(len(v) for v in body_known.values()),
        trajectory_cameras=list(traj_known.keys()),
        skeleton_subjects=list(skel_known.keys()),
        skeleton_profiles=sum(len(v) for v in skel_known.values()),
        pet_subjects=list(pet_known.keys()),
        pet_embeddings=sum(len(v) for v in pet_known.values()),
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
async def reload_all(s: Annotated[AppState, Depends(_get_state)]) -> dict:
    s.recognizer.reload()
    s.body_reid.reload(s.body_db.all_known_by_subject())
    s.trajectory_matcher.reload(s.trajectory_db.known_by_camera())
    s.skeleton_matcher.reload(s.skeleton_db.known_by_subject())
    s.pet_reid.reload(s.pet_db.all_known_by_subject(), s.pet_db.species_by_subject())
    return {
        "face_subjects": s.recognizer.known_subjects(),
        "body_subjects": list(s.body_db.all_known_by_subject().keys()),
        "skeleton_subjects": list(s.skeleton_db.known_by_subject().keys()),
        "pet_subjects": list(s.pet_db.all_known_by_subject().keys()),
    }


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
    embedding = None
    if crop_src.exists():
        crop_bytes = crop_src.read_bytes()
        (known_dir / row.crop_filename).write_bytes(crop_bytes)
        embedding = s.pet_reid.embed(crop_bytes)
    db_id = s.pet_db.add_known(subject, row.species, row.crop_filename, embedding)
    s.pet_db.mark_labeled(pet_id)
    s.pet_reid.reload(s.pet_db.all_known_by_subject(), s.pet_db.species_by_subject())
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


@app.get("/api/trajectories/unknown")
async def list_unknown_trajectories(
    s: Annotated[AppState, Depends(_get_state)], limit: int = 50
) -> list[dict[str, object]]:
    rows = s.trajectory_db.list_unknown(limit=limit)
    return [
        {
            "id": r.id,
            "detected_at": r.detected_at,
            "camera": r.camera,
            "direction_angle": r.direction_angle,
            "speed": round(r.speed, 1),
            "entry_zone": r.entry_zone,
            "exit_zone": r.exit_zone,
        }
        for r in rows
    ]


@app.get("/api/trajectories/subjects")
async def list_trajectory_subjects(
    s: Annotated[AppState, Depends(_get_state)],
) -> dict[str, int]:
    known = s.trajectory_db.known_by_camera()
    counts: dict[str, int] = {}
    for cam_subjects in known.values():
        for subject, vectors in cam_subjects.items():
            counts[subject] = counts.get(subject, 0) + len(vectors)
    return counts


@app.post("/api/trajectories/unknown/{traj_id}/label")
async def label_unknown_trajectory(
    traj_id: int,
    body: dict[str, str],
    s: Annotated[AppState, Depends(_get_state)],
) -> dict[str, str | int]:
    subject = body.get("subject", "").strip()
    if not subject:
        raise HTTPException(status_code=400, detail="subject is required")
    row = s.trajectory_db.get(traj_id)
    if not row:
        raise HTTPException(status_code=404)
    s.trajectory_db.label(traj_id, subject)
    s.trajectory_matcher.reload(s.trajectory_db.known_by_camera())
    return {"id": traj_id, "subject": subject}


@app.delete("/api/trajectories/unknown/{traj_id}")
async def discard_unknown_trajectory(
    traj_id: int, s: Annotated[AppState, Depends(_get_state)]
) -> dict[str, str]:
    if s.trajectory_db.get(traj_id) is None:
        raise HTTPException(status_code=404)
    s.trajectory_db.discard(traj_id)
    return {"status": "discarded"}


@app.get("/api/skeletons/known")
async def list_skeleton_subjects(
    s: Annotated[AppState, Depends(_get_state)],
) -> dict[str, int]:
    known = s.skeleton_db.known_by_subject()
    return {subj: len(vecs) for subj, vecs in known.items()}


@app.get("/api/skeletons/unknown")
async def list_unknown_skeletons(
    s: Annotated[AppState, Depends(_get_state)], limit: int = 50
) -> list[dict[str, object]]:
    rows = s.skeleton_db.list_unknown(limit=limit)
    return [
        {
            "id": r.id,
            "detected_at": r.detected_at,
            "camera": r.camera,
            "shoulder_ratio": round(r.shoulder_ratio, 3),
            "torso_ratio": round(r.torso_ratio, 3),
            "leg_ratio": round(r.leg_ratio, 3),
            "arm_ratio": round(r.arm_ratio, 3),
            "height_px": round(r.height_px),
        }
        for r in rows
    ]


@app.post("/api/skeletons/unknown/{skel_id}/label")
async def label_skeleton(
    skel_id: int,
    body: dict[str, str],
    s: Annotated[AppState, Depends(_get_state)],
) -> dict[str, str | int]:
    subject = body.get("subject", "").strip()
    if not subject:
        raise HTTPException(status_code=400, detail="subject is required")
    row = s.skeleton_db.get(skel_id)
    if not row:
        raise HTTPException(status_code=404)
    s.skeleton_db.label(skel_id, subject)
    s.skeleton_matcher.reload(s.skeleton_db.known_by_subject())
    return {"id": skel_id, "subject": subject}


@app.delete("/api/skeletons/unknown/{skel_id}")
async def discard_skeleton(
    skel_id: int, s: Annotated[AppState, Depends(_get_state)]
) -> dict[str, str]:
    if s.skeleton_db.get(skel_id) is None:
        raise HTTPException(status_code=404)
    s.skeleton_db.discard(skel_id)
    return {"status": "discarded"}


@app.get("/api/protect/events")
async def list_protect_events(
    s: Annotated[AppState, Depends(_get_state)],
    days: int = 7,
    limit: int = 200,
) -> list[dict[str, object]]:
    import time as _time

    now_ms = int(_time.time() * 1000)
    start_ms = now_ms - days * 24 * 3600 * 1000
    events = await s.protect.list_events(
        start_ms, now_ms, types=["smartDetectZone"], limit=limit,
    )
    return [
        {
            "id": ev.get("id"),
            "camera": s.protect.camera_name(ev.get("camera", "")),
            "camera_id": ev.get("camera"),
            "start": ev.get("start"),
            "smart_detect_types": ev.get("smartDetectTypes", []),
            "thumbnail_url": f"/api/protect/events/{ev['id']}/thumbnail",
        }
        for ev in events
        if "person" in ev.get("smartDetectTypes", [])
        and ev.get("id") not in s.analyzed_event_ids
    ]


@app.get("/api/protect/events/{event_id}/thumbnail")
async def protect_event_thumbnail(
    event_id: str, s: Annotated[AppState, Depends(_get_state)]
):
    from fastapi.responses import Response

    thumb = await s.protect.fetch_event_thumbnail(event_id)
    if thumb is None:
        raise HTTPException(status_code=404, detail="thumbnail not found")
    return Response(content=thumb, media_type="image/jpeg")


@app.post("/api/protect/events/{event_id}/analyze-skeleton")
async def analyze_event_skeleton(
    event_id: str,
    body: dict[str, str],
    s: Annotated[AppState, Depends(_get_state)],
) -> dict[str, object]:
    subject = body.get("subject", "").strip()
    if not subject:
        raise HTTPException(status_code=400, detail="subject is required")

    thumb = await s.protect.fetch_event_thumbnail(event_id)
    if thumb is None:
        raise HTTPException(status_code=404, detail="thumbnail not found")

    import numpy as np

    kps = s.skeleton_analyzer.detect_keypoints(thumb)
    if not kps:
        raise HTTPException(status_code=422, detail="no keypoints detected")
    props = s.skeleton_analyzer.extract_proportions(kps[0])
    if props is None:
        raise HTTPException(status_code=422, detail="could not extract proportions")

    vec = np.array(
        [props.shoulder_ratio, props.torso_ratio, props.leg_ratio, props.arm_ratio],
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm

    camera = body.get("camera", "unknown")
    row_id = s.skeleton_db.add(
        camera=camera,
        profile_vector=vec,
        shoulder_ratio=props.shoulder_ratio,
        torso_ratio=props.torso_ratio,
        leg_ratio=props.leg_ratio,
        arm_ratio=props.arm_ratio,
        height_px=props.height_px,
        subject=subject,
    )
    s.skeleton_matcher.reload(s.skeleton_db.known_by_subject())
    s.analyzed_event_ids.add(event_id)
    return {
        "id": row_id,
        "subject": subject,
        "shoulder": round(props.shoulder_ratio, 3),
        "torso": round(props.torso_ratio, 3),
        "leg": round(props.leg_ratio, 3),
        "arm": round(props.arm_ratio, 3),
        "height_px": round(props.height_px),
    }


@app.post("/api/pets/backfill")
async def backfill_pets(
    s: Annotated[AppState, Depends(_get_state)],
    days: int = 30,
    limit: int = 500,
) -> dict[str, int]:
    """Scan Protect event thumbnails with YOLO to find animals Protect missed."""
    import time as _time
    from datetime import UTC, datetime

    now_ms = int(_time.time() * 1000)
    start_ms = now_ms - days * 24 * 3600 * 1000
    events = await s.protect.list_events(
        start_ms, now_ms, types=["smartDetectZone", "motion"], limit=limit,
    )
    log.info("pet_backfill_started", days=days, events=len(events))

    pet_dir = s.settings.data_dir / "unknown_pets"
    pet_dir.mkdir(parents=True, exist_ok=True)
    found = 0
    scanned = 0

    for ev in events:
        event_id = ev.get("id")
        camera_id = ev.get("camera", "")
        if not event_id:
            continue
        thumb = await s.protect.fetch_event_thumbnail(event_id)
        if thumb is None or len(thumb) < 1000:
            continue
        scanned += 1
        try:
            pets = s.pet_detector.detect(thumb)
        except Exception:  # noqa: BLE001
            continue
        if not pets:
            continue

        camera_name = s.protect.camera_name(camera_id)
        ts = datetime.fromtimestamp(ev.get("start", now_ms) / 1000, UTC)
        timestamp = ts.strftime("%Y%m%dT%H%M%S")

        for idx, pet in enumerate(pets):
            crop_bytes = PetDetector.crop(thumb, pet.bbox)
            crop_filename = f"backfill_{timestamp}_{camera_name}_{pet.species}_{idx}_crop.jpg"
            snap_filename = f"backfill_{timestamp}_{camera_name}_{pet.species}_{idx}_snap.jpg"
            (pet_dir / crop_filename).write_bytes(crop_bytes)
            (pet_dir / snap_filename).write_bytes(thumb)
            s.pet_db.add_unknown(
                camera=camera_name,
                species=pet.species,
                confidence=pet.confidence,
                crop_filename=crop_filename,
                snapshot_filename=snap_filename,
                bbox=pet.bbox,
                width_px=pet.width_px,
                height_px=pet.height_px,
            )
            found += 1
            log.info("pet_backfill_found", camera=camera_name, species=pet.species,
                     confidence=round(pet.confidence, 2), event_time=timestamp)

    log.info("pet_backfill_done", scanned=scanned, found=found)
    return {"events_scanned": scanned, "pets_found": found}


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

<h2>Skeleton-validering (Protect-historik)</h2>
<div class="known" id="known-skeletons"></div>
<div style="margin-bottom:1rem">
  <button onclick="loadProtectEvents()">Hämta person-events (7 dagar)</button>
  <span id="event-status" style="color:#888;margin-left:0.5rem"></span>
</div>
<div class="grid" id="grid-events"></div>

<h2>Okända ansikten</h2>
<div class="known" id="known-faces"></div>
<div class="grid" id="grid-faces"></div>

<h2>Rörelsemönster</h2>
<div class="known" id="known-trajectories"></div>
<div class="grid" id="grid-trajectories"></div>

<h2>Okända djur</h2>
<div class="known" id="known-pets"></div>
<div style="margin-bottom:1rem">
  <button onclick="backfillPets()">Sök djur i historik (30 dagar)</button>
  <span id="pet-backfill-status" style="color:#888;margin-left:0.5rem"></span>
</div>
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

  async function loadTrajectories() {
    const r = await fetch('/api/trajectories/unknown');
    const items = await r.json();
    const grid = document.getElementById('grid-trajectories');
    grid.innerHTML = items.length === 0 ? '<p>Inga okända rörelser i kön.</p>' : '';
    for (const t of items) {
      const card = document.createElement('div');
      card.className = 'card';
      const dir = t.direction_angle !== null ? Math.round(t.direction_angle) + '°' : 'stillastående';
      card.innerHTML = `
        <div class="meta">${t.camera} — ${new Date(t.detected_at).toLocaleString('sv-SE')}</div>
        <div class="meta">Riktning: ${dir} — Hastighet: ${t.speed} px/s</div>
        <div class="meta">In: zon ${t.entry_zone} — Ut: zon ${t.exit_zone}</div>
        <div class="row">
          <input type="text" placeholder="Namn (t.ex. Malin)" id="tn-${t.id}">
          <button onclick="labelTraj(${t.id})">Spara</button>
        </div>
        <div class="row"><button class="secondary" onclick="discardTraj(${t.id})">Skippa</button></div>`;
      grid.appendChild(card);
    }
  }
  async function labelTraj(id) {
    const name = document.getElementById('tn-' + id).value.trim();
    if (!name) { alert('Ange namn'); return; }
    const r = await postLabel(`/api/trajectories/unknown/${id}/label`, id, name);
    if (r.ok) await refresh(); else alert('Fel: ' + r.status);
  }
  async function discardTraj(id) {
    await fetch(`/api/trajectories/unknown/${id}`, {method: 'DELETE'});
    await loadTrajectories();
  }

  async function loadProtectEvents() {
    const status = document.getElementById('event-status');
    status.textContent = 'Hämtar events...';
    const r = await fetch('/api/protect/events?days=7&limit=200');
    const events = await r.json();
    const grid = document.getElementById('grid-events');
    grid.innerHTML = '';
    status.textContent = events.length + ' person-events';
    for (const e of events) {
      const card = document.createElement('div');
      card.className = 'card';
      card.id = 'ev-' + e.id;
      const ts = new Date(e.start).toLocaleString('sv-SE');
      card.innerHTML = `
        <img src="${e.thumbnail_url}" alt="event" loading="lazy">
        <div class="meta">${e.camera} — ${ts}</div>
        <div class="row">
          <button onclick="labelEvent('${e.id}','${e.camera}','Malin')" style="background:#16a34a">Malin</button>
          <button onclick="labelEvent('${e.id}','${e.camera}','Madde')" style="background:#2563eb">Madde</button>
          <button onclick="labelEvent('${e.id}','${e.camera}','Loe')" style="background:#9333ea">Loe</button>
        </div>
        <div class="row">
          <input type="text" placeholder="Annat namn" id="en-${e.id}">
          <button onclick="labelEventCustom('${e.id}','${e.camera}')">Spara</button>
          <button class="secondary" onclick="skipEvent('${e.id}')">Skippa</button>
        </div>`;
      grid.appendChild(card);
    }
  }
  async function labelEvent(eventId, camera, name) {
    const card = document.getElementById('ev-' + eventId);
    card.style.opacity = '0.5';
    const r = await fetch('/api/protect/events/' + eventId + '/analyze-skeleton', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({subject: name, camera: camera}),
    });
    if (r.ok) {
      card.remove();
      loadKnown('/api/skeletons/known', document.getElementById('known-skeletons'), 'Skeleton-profiler');
    } else {
      card.style.opacity = '1';
      const err = await r.json().catch(() => ({}));
      card.querySelector('.meta').insertAdjacentHTML('afterend',
        '<div style="color:#f87171;font-size:0.85rem">✗ ' + (err.detail || r.status) + '</div>');
    }
  }
  async function labelEventCustom(eventId, camera) {
    const name = document.getElementById('en-' + eventId).value.trim();
    if (!name) { alert('Ange namn'); return; }
    await labelEvent(eventId, camera, name);
  }
  function skipEvent(eventId) {
    const card = document.getElementById('ev-' + eventId);
    card.remove();
  }

  async function backfillPets() {
    const status = document.getElementById('pet-backfill-status');
    status.textContent = 'Söker djur i gamla events...';
    const r = await fetch('/api/pets/backfill?days=30&limit=500', {method: 'POST'});
    const d = await r.json();
    status.textContent = `Klart: ${d.pets_found} djur hittade i ${d.events_scanned} events`;
    await loadPets();
  }

  async function refresh() {
    await Promise.all([
      loadKnown('/api/subjects', document.getElementById('known-faces'), 'Tränade ansikten'),
      loadKnown('/api/pets/subjects', document.getElementById('known-pets'), 'Tränade djur'),
      loadKnown('/api/bodies/subjects', document.getElementById('known-bodies'), 'Tränade personer'),
      loadKnown('/api/trajectories/subjects', document.getElementById('known-trajectories'), 'Tränade rörelser'),
      loadKnown('/api/skeletons/known', document.getElementById('known-skeletons'), 'Skeleton-profiler'),
      loadFaces(),
      loadPets(),
      loadBodies(),
      loadTrajectories(),
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
