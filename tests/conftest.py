"""Configuração pytest — adiciona src/ ao path."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
