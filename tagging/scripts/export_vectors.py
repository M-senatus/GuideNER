"""Thin wrapper for vector export."""

from __future__ import annotations

import sys
from pathlib import Path


TAGGING_ROOT = Path(__file__).resolve().parents[1]
if str(TAGGING_ROOT) not in sys.path:
    sys.path.insert(0, str(TAGGING_ROOT))

from src.infer.export_hidden_states import main


if __name__ == "__main__":
    main()
