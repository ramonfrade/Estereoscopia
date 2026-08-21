"""Testes do payload do worker: o erro de calibração deve acompanhar o frame."""

import numpy as np

import worker


def _fake_result() -> dict:
    return {
        "display_left": np.zeros((20, 30, 3), dtype=np.uint8),
        "display_right": np.zeros((20, 30, 3), dtype=np.uint8),
        "disparity": np.zeros((20, 30), dtype=np.float32),
        "depth_cm": 42.0,
        "disparity_px": 5.0,
        "sample_point": {"x": 15, "y": 10},
        "track_mode": "closest",
        "depth_mode": "sgbm",
    }


def test_payload_includes_calibration_error():
    payload = worker.build_frame_payload(_fake_result(), calibration_error_pct=1.23)
    assert payload["calibration_error_pct"] == 1.23


def test_payload_calibration_error_none_by_default():
    payload = worker.build_frame_payload(_fake_result())
    assert payload["calibration_error_pct"] is None


def test_payload_keeps_core_fields():
    payload = worker.build_frame_payload(_fake_result())
    assert payload["depth_cm"] == 42.0
    assert payload["disparity_px"] == 5.0
    assert payload["track_mode"] == "closest"
    assert set(payload["images"]) == {"left", "right", "disparity"}
