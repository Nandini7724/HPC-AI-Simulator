"""
amdahl.py
---------
AMDAHL'S LAW

    Speedup(N) = 1 / ((1 - P) + (P / N))

where P is the parallelizable fraction of the workload and N is the
number of processing units. Demonstrates that speedup is bounded by the
non-parallelizable (sequential) fraction, no matter how many processors
are added.
"""


def amdahl_speedup(parallel_fraction: float, processors: int) -> float:
    p = min(max(parallel_fraction, 0.0), 1.0)
    n = max(processors, 1)
    denom = (1 - p) + (p / n)
    if denom <= 0:
        return float(n)
    return 1.0 / denom


def amdahl_curve(parallel_fraction: float, processor_points=None):
    """Return [(N, speedup), ...] for a standard set of processor counts."""
    if processor_points is None:
        processor_points = [1, 2, 4, 8, 16, 32, 64]
    return [{"processors": n, "speedup": amdahl_speedup(parallel_fraction, n)}
            for n in processor_points]
