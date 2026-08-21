"""Testes do detector de objetos em fundo homogêneo e conversão px->cm."""

import cv2
import numpy as np
import pytest

from object_detector import HomogeneousBgDetector, object_size_cm


def _homogeneous_image(rect: tuple[int, int, int, int] | None = None, specks: int = 0) -> np.ndarray:
    img = np.full((300, 400), 255, dtype=np.uint8)
    if rect is not None:
        x, y, w, h = rect
        img[y : y + h, x : x + w] = 0
    rng = np.random.default_rng(42)
    for _ in range(specks):
        x, y = rng.integers(0, 400), rng.integers(0, 300)
        img[y : y + 3, x : x + 3] = 0
    return img


def test_detect_objects_finds_rectangle():
    detector = HomogeneousBgDetector()
    contours = detector.detect_objects(_homogeneous_image(rect=(150, 100, 100, 80)))
    assert len(contours) == 1
    # boundingRect conta pixels: retângulo 100x80 exatamente
    assert cv2.boundingRect(contours[0]) == (150, 100, 100, 80)


def test_detect_objects_ignores_small_specks():
    detector = HomogeneousBgDetector()
    contours = detector.detect_objects(
        _homogeneous_image(rect=(150, 100, 100, 80), specks=8)
    )
    assert len(contours) == 1


def test_detect_objects_none_on_uniform_background():
    detector = HomogeneousBgDetector()
    contours = detector.detect_objects(_homogeneous_image())
    assert contours == []


def test_object_size_cm():
    rect = (150, 100, 100, 80)
    img = _homogeneous_image(rect=rect)
    detector = HomogeneousBgDetector()
    (cnt,) = detector.detect_objects(img)
    w_cm, h_cm = object_size_cm(cnt, pixel_cm_ratio=10.0)
    # minAreaRect pode trocar largura/altura conforme a rotação; mede os dois
    # lados (100x80 px). Erro de ~1px é inerente à medição por contorno.
    measured = sorted((w_cm, h_cm))
    expected = sorted((10.0, 8.0))
    assert measured[0] == pytest.approx(expected[0], rel=0.05)
    assert measured[1] == pytest.approx(expected[1], rel=0.05)
