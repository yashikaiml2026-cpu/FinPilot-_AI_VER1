from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def log_event(event_type: str, message: str, payload: dict[str, Any] | None = None) -> None:
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "finpilot.log"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "message": message,
        "payload": payload or {},
    }
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")
