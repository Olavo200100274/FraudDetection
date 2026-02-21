from __future__ import annotations
import json
import platform
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional

@dataclass
class RunTimer:
    start: float
    end: Optional[float] = None

    def stop(self) -> float:
        self.end = time.perf_counter()
        return float(self.end - self.start)

def start_timer() -> RunTimer:
    return RunTimer(start=time.perf_counter())

def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def build_system_meta() -> Dict[str, Any]:
    return {
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }