"""Log each pipeline step as one JSON line in logs/<YYYY-MM-DD>.jsonl: run id,
step, inputs, outputs, latency and any error. Use one RunLogger per question."""

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]  # repo root
LOG_DIR = ROOT / "logs"  # gitignored


def to_json(value: Any) -> Any:
    """Make values json.dumps can't handle readable: models, dates, Decimals."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return str(value)


class RunLogger:
    """Writes the steps of one run, all sharing one run_id."""

    def __init__(self, run_id: str | None = None, log_dir: Path = LOG_DIR) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.log_dir = log_dir

    @contextmanager
    def step(self, name: str, **inputs: Any) -> Iterator[dict[str, Any]]:
        """Time a step; the caller fills the yielded dict with its outputs."""
        started = datetime.now(UTC)
        t0 = time.perf_counter()
        outputs: dict[str, Any] = {}
        error = None
        try:
            yield outputs
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.write(
                {
                    "run_id": self.run_id,
                    "step": name,
                    "started": started.isoformat(),
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "inputs": inputs,
                    "outputs": outputs,
                    "error": error,
                }
            )

    def write(self, record: dict[str, Any]) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"{record['started'][:10]}.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(record, default=to_json) + "\n")


def read_log(path: Path) -> list[dict[str, Any]]:
    """Read a log file back as a list of records."""
    return [json.loads(line) for line in path.open() if line.strip()]
