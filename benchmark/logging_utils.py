"""Logging helpers for benchmark runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class JsonlWriter:
    """Simple JSONL writer with one record per line."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, sort_keys=True))
        self._fh.write("\n")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self.close()


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON file with stable key ordering."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def make_run_dir(logdir: Path, game: str, seed: int) -> Path:
    """Create a unique run directory under `logdir`."""
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = f"{game}_seed{seed}_{stamp}"
    run_dir = logdir / base_name
    suffix = 1
    while run_dir.exists():
        run_dir = logdir / f"{base_name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
