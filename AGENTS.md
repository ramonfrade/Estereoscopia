# AGENTS.md — estereoscopia

## Project intent

Stereo vision object detection + distance estimation using two Logitech C270 webcams on Linux, Python, and OpenCV.

## Current state

Active code lives in `src/` (Python worker + engine) and `web/` (TypeScript frontend). **`src_old/` is frozen** — it works as the reference implementation; do not modify it.

Depth estimation default: **MediaPipe face detection + triangulation** (same approach as `src_old/stereoVision.py`). Optional `--depth-mode sgbm` uses dense disparity.

## Known hardware

- 2× Logitech C270 (USB ID `046d:0825`)
- Detected on `/dev/video*` (likely `video0`, `video1`)

## Toolchain (inferred from constraints)

- Language: **Python 3**
- Library: **OpenCV** (`cv2`)
- Platform: **Linux only**
- Tests: `pytest` in `tests/`

## Camera setup notes

- `v4l2-ctl` for querying/resetting camera properties
- Stereo calibration (chessboard + `cv2.stereoCalibrate`) will be needed
- Depth from face triangulation (default) or disparity map via `cv2.StereoSGBM_create`

## Future-proofing

When adding tools or config, prefer conventional Python project layout:
- `src/` or top-level module package
- `requirements.txt` or `pyproject.toml`
- `tests/` with `pytest`
- `.gitignore` covering `__pycache__/`, `*.pyc`, `.venv/`, calibration files

## Git workflow (user rule — memorize, always follow)

- **Before starting ANY task**: make a commit of the current working tree
  (so the baseline is recorded). If the tree is already clean, nothing to do.
- **After finishing ANY task**: make a commit of the completed work.
- This applies to every task, without exceptions.

