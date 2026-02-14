import time
from contextlib import contextmanager

@contextmanager
def timed(section: str, timings: list):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    timings.append((section, dt))

def fmt_seconds(s: float) -> str:
    if s < 1:
        return f"~{s:.1f}s"
    if s < 60:
        return f"~{s:.1f}s"
    m = int(s // 60)
    r = s - 60 * m
    return f"~{m}m {r:.0f}s"
