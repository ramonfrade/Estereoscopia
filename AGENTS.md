# AGENTS.md — estereoscopia

## Project intent

Stereo vision object detection + distance estimation using two Logitech C270 webcams on Linux, Python, and OpenCV.

## Current state

Greenfield — only `objetivos.txt` exists with the project brief. No source code, no manifests, no dependencies declared yet.

## Known hardware

- 2× Logitech C270 (USB ID `046d:0825`)
- Detected on `/dev/video*` (likely `video0`, `video1`)

## Toolchain (inferred from constraints)

- Language: **Python 3**
- Library: **OpenCV** (`cv2`)
- Platform: **Linux only**
- No existing build/test/lint config — these need to be introduced as the project grows.

## Camera setup notes

- `v4l2-ctl` for querying/resetting camera properties
- Stereo calibration (chessboard + `cv2.stereoCalibrate`) will be needed
- Depth from disparity map via `cv2.StereoSGBM_create`

## Future-proofing

When adding tools or config, prefer conventional Python project layout:
- `src/` or top-level module package
- `requirements.txt` or `pyproject.toml`
- `tests/` with `pytest`
- `.gitignore` covering `__pycache__/`, `*.pyc`, `.venv/`, calibration files
