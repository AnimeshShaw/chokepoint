"""CLI wrapper for :mod:`chokepoint.analysis.statistics`.

    python analysis/statistics.py --results-root results/runs
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chokepoint.analysis.statistics import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
