"""
cpu_model.py
------------
CPU PERFORMANCE MODEL

More cores increase parallelism, but synchronization, memory access and
scheduling overhead prevent perfect linear scaling. We model this with a
smooth efficiency-decay function of core count.
"""
from dataclasses import dataclass
from . import config


@dataclass
class CPUResult:
    cores: int
    efficiency: float          # 0..1, "how well the cores are used"
    simd_enabled: bool
    effective_gflops: float    # usable compute throughput


def compute_cpu_performance(cores: int, simd_enabled: bool = False) -> CPUResult:
    cores = max(int(cores), 1)

    # CPU Efficiency = f(number of cores): gradually decreases as more
    # cores are added, representing synchronization / memory-bus overhead.
    efficiency = 1.0 / (1.0 + config.CPU_OVERHEAD_COEFF * (cores - 1))
    efficiency = max(efficiency, config.CPU_EFFICIENCY_FLOOR)

    vector_factor = config.VECTOR_ACCEL_FACTOR if simd_enabled else 1.0

    # Effective CPU Performance = Base x Cores x Efficiency x SIMD factor
    effective_gflops = (config.BASE_CPU_GFLOPS_PER_CORE * cores
                         * efficiency * vector_factor)

    return CPUResult(
        cores=cores,
        efficiency=efficiency,
        simd_enabled=simd_enabled,
        effective_gflops=effective_gflops,
    )
