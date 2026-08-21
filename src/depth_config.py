"""Parâmetros de profundidade — alinhados ao src_old e calibration.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CALIBRATION_JSON = PROJECT_ROOT / "calibration.json"

# Valores padrão (B=15 cm — distância entre centros das lentes, f=8, alpha=56.6)
DEFAULT_BASELINE_CM = 15.0
DEFAULT_FOCAL_MM = 8.0
DEFAULT_FOV_DEG = 56.6


class DepthMode(str, Enum):
    FACE = "face"
    SGBM = "sgbm"


@dataclass(frozen=True)
class DepthConfig:
    baseline_cm: float = DEFAULT_BASELINE_CM
    focal_mm: float = DEFAULT_FOCAL_MM
    fov_deg: float = DEFAULT_FOV_DEG
    face_min_confidence: float = 0.7
    calibration_error_pct: float | None = None

    @classmethod
    def from_calibration_json(cls, path: Path = CALIBRATION_JSON) -> DepthConfig:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        err = data.get("calibration_error_pct")
        return cls(
            baseline_cm=float(data.get("physical_baseline_cm", DEFAULT_BASELINE_CM)),
            focal_mm=DEFAULT_FOCAL_MM,
            fov_deg=DEFAULT_FOV_DEG,
            calibration_error_pct=float(err) if err is not None else None,
        )
