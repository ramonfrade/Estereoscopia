"""Testes do salvamento dos parâmetros de reconhecimento do tabuleiro."""

import json

from detect_chessboard import save_recognition_params


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
