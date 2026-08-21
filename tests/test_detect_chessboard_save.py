"""Testes do salvamento e acumulação dos parâmetros de reconhecimento."""

import json
import math

import pytest

from detect_chessboard import (
    DetectionStats,
    build_diff_text,
    save_recognition_params,
)


def test_save_writes_pattern_and_preserves_existing(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"Q": [[1.0]], "frameSize": [640, 480]}), encoding="utf-8")

    result = save_recognition_params((7, 7), path=path)

    assert result["chessboardSize"] == [7, 7]
    assert result["pattern"] == "8x8"
    assert result["Q"] == [[1.0]]  # campos existentes preservados
    assert json.loads(path.read_text(encoding="utf-8")) == result


def test_save_creates_file_when_missing(tmp_path):
    path = tmp_path / "calibration.json"

    save_recognition_params((7, 3), path=path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["chessboardSize"] == [7, 3]
    assert data["pattern"] == "8x4"


def test_save_with_board_size_derives_square(tmp_path):
    path = tmp_path / "calibration.json"

    save_recognition_params((7, 7), board_size_cm=(34.624, 34.624), path=path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["board_size_cm"] == [34.624, 34.624]
    assert data["square_cm"] == 4.328


def test_save_without_board_keeps_existing_values(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps({"board_size_cm": [34.624, 34.624], "square_cm": 4.328}),
        encoding="utf-8",
    )

    save_recognition_params((7, 7), path=path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["board_size_cm"] == [34.624, 34.624]
    assert data["square_cm"] == 4.328


def test_save_with_error_writes_calibration_error_pct(tmp_path):
    path = tmp_path / "calibration.json"

    save_recognition_params((7, 7), calibration_error_pct=0.6999, path=path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["calibration_error_pct"] == 0.7


def test_stats_accumulate_and_average():
    stats = DetectionStats(configured=(7, 7))
    stats.record((7, 7), 0.5)
    stats.record((7, 7), 0.7)
    stats.record((7, 3), 0.9)
    stats.record(None, None)  # quadro sem detecção

    assert stats.frames == 4
    assert stats.detections == 3
    assert stats.mean_error == pytest.approx(0.7)
    assert stats.min_error == pytest.approx(0.5)
    assert stats.max_error == pytest.approx(0.9)
    assert stats.dominant_pattern == (7, 7)
    assert stats.pattern_counts == {(7, 7): 2, (7, 3): 1}


def test_stats_dominant_falls_back_to_configured():
    stats = DetectionStats(configured=(7, 7))
    assert stats.dominant_pattern == (7, 7)
    assert stats.mean_error is None


def test_stats_ignores_non_finite_errors():
    stats = DetectionStats()
    stats.record((7, 7), float("inf"))
    stats.record((7, 7), math.nan)

    assert stats.detections == 2
    assert stats.errors == []
    assert stats.mean_error is None


def test_diff_text_shows_changes():
    previous = {
        "chessboardSize": [7, 7],
        "pattern": "8x8",
        "square_cm": 4.4,
        "calibration_error_pct": 1.2,
    }
    text = build_diff_text(
        previous, (7, 7), board_size_cm=(34.624, 34.624), mean_error=0.45
    )
    assert "chessboardSize" in text
    assert "4.4" in text
    assert "4.328" in text
    assert "0.45" in text
    assert "1.2" in text


def test_diff_text_marks_unchanged():
    previous = {"chessboardSize": [7, 7], "pattern": "8x8"}
    text = build_diff_text(previous, (7, 7))
    assert "inalterado" in text
    assert "->" not in text
