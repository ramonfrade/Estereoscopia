"""Detecção de rosto via MediaPipe — encapsula a tecnologia usada em src_old."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import mediapipe as mp


@dataclass(frozen=True)
class FaceCenter:
    x: float
    y: float
    score: float
    bbox: tuple[int, int, int, int]


def bbox_center(
    xmin: float, ymin: float, width: float, height: float, frame_w: int, frame_h: int
) -> tuple[float, float, tuple[int, int, int, int]]:
    bx = int(xmin * frame_w)
    by = int(ymin * frame_h)
    bw = int(width * frame_w)
    bh = int(height * frame_h)
    cx = bx + bw / 2
    cy = by + bh / 2
    return cx, cy, (bx, by, bw, bh)


class FaceDetector:
    """Wrapper fino sobre MediaPipe Face Detection (mesma API do src_old)."""

    def __init__(self, min_confidence: float = 0.7):
        import mediapipe as mp  # lazy: evita PortAudio em ambientes headless/CI

        self._mp_face = mp.solutions.face_detection
        self._draw = mp.solutions.drawing_utils
        self._detector = self._mp_face.FaceDetection(min_detection_confidence=min_confidence)

    def detect_primary(self, frame_bgr) -> FaceCenter | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        if not results.detections:
            return None

        detection = results.detections[0]
        h, w = frame_bgr.shape[:2]
        box = detection.location_data.relative_bounding_box
        cx, cy, bbox = bbox_center(box.xmin, box.ymin, box.width, box.height, w, h)
        score = float(detection.score[0]) if detection.score else 0.0
        return FaceCenter(x=cx, y=cy, score=score, bbox=bbox)

    def process_with_draw(self, frame_bgr) -> tuple[FaceCenter | None, object | None]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        if not results.detections:
            return None, None

        detection = results.detections[0]
        self._draw.draw_detection(frame_bgr, detection)
        h, w = frame_bgr.shape[:2]
        box = detection.location_data.relative_bounding_box
        cx, cy, bbox = bbox_center(box.xmin, box.ymin, box.width, box.height, w, h)
        score = float(detection.score[0]) if detection.score else 0.0
        return FaceCenter(x=cx, y=cy, score=score, bbox=bbox), detection

    def detect_all(self, frame_bgr) -> list[FaceCenter]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)
        if not results.detections:
            return []

        h, w = frame_bgr.shape[:2]
        faces: list[FaceCenter] = []
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            cx, cy, bbox = bbox_center(box.xmin, box.ymin, box.width, box.height, w, h)
            score = float(detection.score[0]) if detection.score else 0.0
            faces.append(FaceCenter(x=cx, y=cy, score=score, bbox=bbox))
        return faces

    def detect_all_with_draw(self, frame_bgr) -> list[FaceCenter]:
        """Detecta TODOS os rostos e desenha um retângulo em cada um.

        Retorna a lista de FaceCenter (uma por rosto), na ordem das detecções.
        """
        faces = self.detect_all(frame_bgr)
        for face in faces:
            bx, by, bw, bh = face.bbox
            cv2.rectangle(frame_bgr, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        return faces

    def close(self) -> None:
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
