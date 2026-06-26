"""
Demonstração estéreo em tempo real (Linux).

Substitui o stereoVision.py original (Windows/DSHOW + MediaPipe).
Usa stereo_engine + calibração local.

  python src/stereoVision.py
"""

import stereo_engine

if __name__ == "__main__":
    from worker import run_demo

    run_demo()
