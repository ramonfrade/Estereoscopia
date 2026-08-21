"""Detecção autônoma de tabuleiro de xadrez (8x8 ou 8x4) e erro percentual.

Fluxo:
  detect_chessboard_auto(gray)  -> (size, corners) | None
  board_error_pct(gray, corners, size) -> erro % (2 casas ao exibir)
  square_size_cm(width_cm, height_cm, size) -> tamanho do quadrado em cm
"""

from __future__ import annotations

import cv2
import numpy as np

# Cantos internos: (7,7) = tabuleiro 8x8; (7,3) = meio tabuleiro 8x4.
DEFAULT_CANDIDATES: tuple[tuple[int, int], ...] = ((7, 7), (7, 3))

FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Tolerâncias da validação geométrica pós-detecção.
MIN_COVERAGE = 0.03  # área do padrão >= 3% da imagem
RATIO_TOLERANCE = (0.55, 1.8)  # razão de aspecto medida x esperada


def _ordered_pts(corners: np.ndarray) -> np.ndarray:
    """Cantos em (N,2) float64, na ordem linha-por-linha do findChessboardCorners."""
    return corners.reshape(-1, 2).astype(np.float64)


def _expected_ratio(size: tuple[int, int]) -> float:
    """Razão de aspecto esperada (largura/altura em quadrados): 8x8 -> 1, 8x4 -> 2."""
    cols, rows = size
    return (cols + 1) / (rows + 1)


def _plausible(corners: np.ndarray, size: tuple[int, int], image_area: int) -> bool:
    """Descarta detecções espúrias via razão de aspecto e cobertura mínima."""
    pts = _ordered_pts(corners)
    cols, rows = size
    if pts.shape[0] != cols * rows:
        return False

    # Ordem linha-por-linha: topo-esquerda = 0, topo-direita = cols-1,
    # base-esquerda = (rows-1)*cols.
    tl, tr = pts[0], pts[cols - 1]
    bl = pts[(rows - 1) * cols]
    w_px = float(np.linalg.norm(tr - tl))
    h_px = float(np.linalg.norm(bl - tl))
    if min(w_px, h_px) <= 0:
        return False

    ratio = max(w_px, h_px) / min(w_px, h_px)
    expected = _expected_ratio(size)
    lo, hi = RATIO_TOLERANCE
    if ratio < expected * lo or ratio > expected * hi:
        return False

    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    covered = (xmax - xmin) * (ymax - ymin) / max(1, image_area)
    return covered >= MIN_COVERAGE


def detect_chessboard_auto(
    gray: np.ndarray,
    candidates: tuple[tuple[int, int], ...] = DEFAULT_CANDIDATES,
) -> tuple[tuple[int, int], np.ndarray] | None:
    """Tenta cada padrão em ordem decrescente; retorna (size, corners) ou None.

    A ordem importa: um (7,3) pode ser detectado como subgrade de um 8x8
    inteiro; tentar (7,7) primeiro evita classificar o tabuleiro cheio
    como "metade".
    """
    image_area = gray.shape[0] * gray.shape[1]
    for size in candidates:
        found, corners = cv2.findChessboardCorners(gray, size, None, FIND_FLAGS)
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        if _plausible(corners, size, image_area):
            return size, corners
    return None


def board_error_pct(gray: np.ndarray, corners: np.ndarray, size: tuple[int, int]) -> float:
    """Erro percentual com base no tamanho médio do quadrado (em %).

    Ajusta uma homografia do grid ideal (quadrados unitários) aos cantos
    detectados e divide o RMS da reprojeção (px) pelo tamanho médio do
    quadrado em px. Independe de intrínsecos da câmera.
    """
    pts = _ordered_pts(corners)
    cols, rows = size
    n = cols * rows
    if pts.shape[0] != n:
        return float("inf")

    ideal = np.zeros((n, 2), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            ideal[r * cols + c] = (c, r)

    homography, _ = cv2.findHomography(ideal, pts)
    if homography is None:
        return float("inf")
    projected = cv2.perspectiveTransform(ideal.reshape(-1, 1, 2), homography).reshape(-1, 2)
    rms_px = float(np.sqrt(np.mean(np.sum((projected - pts) ** 2, axis=1))))

    # Tamanho médio do quadrado em px: distâncias entre cantos adjacentes.
    edges: list[float] = []
    for r in range(rows):
        for c in range(cols - 1):
            i = r * cols + c
            edges.append(float(np.linalg.norm(pts[i + 1] - pts[i])))
    for r in range(rows - 1):
        for c in range(cols):
            i = r * cols + c
            edges.append(float(np.linalg.norm(pts[i + cols] - pts[i])))

    square_px = float(np.mean(edges)) if edges else 0.0
    if square_px <= 0:
        return float("inf")
    return rms_px / square_px * 100.0


def square_size_cm(
    board_width_cm: float, board_height_cm: float, size: tuple[int, int]
) -> float:
    """Tamanho do quadrado em cm a partir das dimensões externas do tabuleiro.

    size são cantos internos, logo o tabuleiro tem (cols+1) x (rows+1)
    quadrados; a média das duas direções absorve imprecisão de medição.
    """
    cols, rows = size
    if cols < 0 or rows < 0:
        raise ValueError("Tamanho do padrão inválido.")
    return (board_width_cm / (cols + 1) + board_height_cm / (rows + 1)) / 2.0
