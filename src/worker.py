"""
Worker Python invocável sob demanda pelo TypeScript.

Modos:
  serve  — API HTTP (FastAPI) para o frontend web
  once   — captura um frame e imprime JSON no stdout (para spawn via child_process)
  demo   — loop OpenCV local (substitui stereoVision.py no Linux)

Uso:
  python src/worker.py serve --port 8000
  python src/worker.py once
  python src/worker.py demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Garante imports quando executado da raiz do projeto
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from depth_config import DepthMode
from stereo_engine import (
    StereoEngine,
    annotate_frame,
    disparity_to_colormap,
    encode_image_b64,
)

engine: StereoEngine | None = None
default_depth_mode = DepthMode.FACE


class SamplePoint(BaseModel):
    x: int | None = None
    y: int | None = None


def _get_engine() -> StereoEngine:
    if engine is None:
        raise RuntimeError("Motor estéreo não inicializado.")
    return engine


def build_frame_payload(result: dict, fps: float | None = None) -> dict:
    point = result["sample_point"]
    depth_cm = result["depth_cm"]
    track_mode = result.get("track_mode", "")
    left = annotate_frame(result["display_left"], depth_cm, point, track_mode)
    right = annotate_frame(result["display_right"], depth_cm, point, track_mode)
    disp_vis = disparity_to_colormap(result["disparity"])

    payload = {
        "depth_cm": depth_cm,
        "disparity_px": result["disparity_px"],
        "sample_point": point,
        "track_mode": track_mode,
        "depth_mode": result.get("depth_mode", default_depth_mode.value),
        "images": {
            "left": encode_image_b64(left),
            "right": encode_image_b64(right),
            "disparity": encode_image_b64(disp_vis),
        },
    }
    if fps is not None:
        payload["fps"] = round(fps, 1)
    return payload


def run_once(depth_mode: DepthMode = default_depth_mode) -> dict:
    eng = StereoEngine(depth_mode=depth_mode)
    eng.open_cameras()
    try:
        result = eng.process_frame()
        return build_frame_payload(result)
    finally:
        eng.close_cameras()


def run_demo(depth_mode: DepthMode = default_depth_mode) -> None:
    eng = StereoEngine(depth_mode=depth_mode)
    eng.open_cameras()
    print("Demo local — pressione 'q' para sair.")
    try:
        while True:
            start = time.time()
            result = eng.process_frame()
            point = result["sample_point"]
            depth_cm = result["depth_cm"]

            left = annotate_frame(result["display_left"], depth_cm, point, result.get("track_mode", ""))
            right = annotate_frame(result["display_right"], depth_cm, point, result.get("track_mode", ""))
            disp_vis = disparity_to_colormap(result["disparity"])

            cv2.imshow("Esquerda", left)
            cv2.imshow("Direita", right)
            cv2.imshow("Disparidade", disp_vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            elapsed = time.time() - start
            if elapsed > 0:
                print(f"FPS: {1 / elapsed:.1f} | Distância: {depth_cm}")
    finally:
        eng.close_cameras()
        cv2.destroyAllWindows()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = StereoEngine(depth_mode=default_depth_mode)
    engine.open_cameras()
    yield
    engine.close_cameras()
    engine = None


app = FastAPI(title="Estereoscopia Worker", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    eng = _get_engine()
    layout = eng.layout
    assert layout is not None
    return {
        "cameras_open": bool(eng.caps),
        "calib_left_index": layout.calib_left,
        "calib_right_index": layout.calib_right,
        "physical_left_index": layout.physical_left,
        "physical_right_index": layout.physical_right,
        "frame_size": list(eng.frame_size),
        "depth_mode": eng.depth_mode.value,
    }


@app.get("/frame")
def get_frame(x: int | None = None, y: int | None = None):
    try:
        eng = _get_engine()
        start = time.time()
        result = eng.process_frame(sample_x=x, sample_y=y)
        fps = 1.0 / (time.time() - start) if time.time() > start else None
        return build_frame_payload(result, fps=fps)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/frame")
def post_frame(point: SamplePoint):
    return get_frame(x=point.x, y=point.y)


def main() -> None:
    global default_depth_mode
    parser = argparse.ArgumentParser(description="Worker estéreo Python")
    parser.add_argument(
        "mode",
        choices=["serve", "once", "demo"],
        nargs="?",
        default="serve",
        help="serve=API HTTP | once=JSON no stdout | demo=janelas OpenCV",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--depth-mode",
        choices=[m.value for m in DepthMode],
        default=default_depth_mode.value,
        help="face=MediaPipe+triangulação (padrão, como src_old) | sgbm=disparidade densa",
    )
    args = parser.parse_args()

    default_depth_mode = DepthMode(args.depth_mode)

    if args.mode == "once":
        print(json.dumps(run_once(depth_mode=default_depth_mode)))
        return

    if args.mode == "demo":
        run_demo(depth_mode=default_depth_mode)
        return

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
