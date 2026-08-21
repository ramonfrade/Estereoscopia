"""Testes do módulo chessboard_auto: detecção autônoma 8x8 / 8x4 e erro percentual."""

import cv2
import numpy as np
import pytest

from chessboard_auto import (
    board_error_pct,
    detect_chessboard_auto,
    square_size_cm,
)

FULL_SIZE = (7, 7)  # tabuleiro 8x8 quadrados
HALF_SIZE = (7, 3)  # meio tabuleiro 8x4 quadrados


def make_board(cols: int, rows: int, square: int = 40) -> np.ndarray:
    """Gera imagem em tons de cinza de um tabuleiro cols x rows quadrados."""
    h, w = rows * square, cols * square
    img = np.full((h, w), 255, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                img[r * square : (r + 1) * square, c * square : (c + 1) * square] = 0
    return img


def test_detect_full_board_8x8():
    gray = make_board(8, 8)
    result = detect_chessboard_auto(gray)
    assert result is not None
    size, corners = result
    assert size == FULL_SIZE
    assert corners.reshape(-1, 2).shape[0] == 7 * 7


def test_detect_half_board_8x4():
    gray = make_board(8, 4)
    result = detect_chessboard_auto(gray)
    assert result is not None
    size, corners = result
    assert size == HALF_SIZE
    assert corners.reshape(-1, 2).shape[0] == 7 * 3


def test_full_board_is_not_classified_as_half():
    gray = make_board(8, 8)
    size, _ = detect_chessboard_auto(gray)
    assert size == FULL_SIZE


def test_error_near_zero_on_perfect_board():
    gray = make_board(8, 8)
    size, corners = detect_chessboard_auto(gray)
    err = board_error_pct(gray, corners, size)
    assert 0.0 <= err < 1.0


def test_error_near_zero_on_half_board():
    gray = make_board(8, 4)
    size, corners = detect_chessboard_auto(gray)
    err = board_error_pct(gray, corners, size)
    assert 0.0 <= err < 1.0


def test_detect_with_mild_perspective():
    gray = make_board(8, 8, square=50)
    h, w = gray.shape
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [w * 0.05, h * 0.03],
            [w * 0.95, h * 0.07],
            [w * 0.91, h * 0.97],
            [w * 0.09, h * 0.93],
        ]
    )
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(gray, m, (w, h), borderValue=255)
    result = detect_chessboard_auto(warped)
    assert result is not None
    size, corners = result
    assert size == FULL_SIZE
    err = board_error_pct(warped, corners, size)
    assert 0.0 <= err < 2.0


def test_square_size_cm():
    # 8x4 (meio tabuleiro): largura 80 / 8 colunas = 10; altura 40 / 4 linhas = 10
    assert square_size_cm(80.0, 40.0, HALF_SIZE) == pytest.approx(10.0)
    # 8x8: 80 / 8 = 10; 40 / 8 = 5 -> média 7.5
    assert square_size_cm(80.0, 40.0, FULL_SIZE) == pytest.approx(7.5)
