from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CameraConfig(BaseModel):
    alert_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    family_zone: bool = False
    always_alert_objects: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    unifi_host: str = "192.168.0.10"
    unifi_user: str
    unifi_pass: str
    unifi_verify_tls: bool = False

    ntfy_url: str = "https://ntfy.sh"
    ntfy_topic: str
    ntfy_token: str | None = None

    data_dir: Path = Path("/data")
    cameras_config_path: Path = Path("/config/cameras.yaml")
    family_macs_path: Path = Path("/config/family_macs.yaml")

    face_tolerance: float = 0.6
    min_face_width_px: int = 60
    alert_score_threshold: float = 0.6
    body_similarity_threshold: float = 0.65

    log_level: str = "INFO"
    bind_host: str = "0.0.0.0"
    bind_port: int = 8000


def load_cameras(path: Path) -> dict[str, CameraConfig]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text())
    cameras_raw = raw.get("cameras", {})
    return {name: CameraConfig(**cfg) for name, cfg in cameras_raw.items()}


def load_family_macs(path: Path) -> dict[str, str]:
    """Return mapping of MAC -> family member name.

    YAML schema accepts either a single MAC string or a list per person:
        members:
          Malin: "aa:bb:..."
          Loe:
            - "11:22:..."
            - "33:44:..."   # alt MAC after randomization rotation
    """
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text())
    out: dict[str, str] = {}
    for name, value in (raw.get("members") or {}).items():
        if isinstance(value, str):
            out[value.lower()] = name
        elif isinstance(value, list):
            for mac in value:
                if isinstance(mac, str):
                    out[mac.lower()] = name
    return out
