from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    from apps.exp001_streamlit import main as run_movement_lab

    run_movement_lab()


if __name__ == "__main__":
    main()
