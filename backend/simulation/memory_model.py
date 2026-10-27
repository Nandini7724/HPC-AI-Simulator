
from dataclasses import dataclass

from . import config


@dataclass
class MemoryResult:
    required_gb: float
    available_gb: float
    utilization_pct: float
    fits: bool
    penalty_factor: float


def evaluate_memory(
    memory_required_gb: float,
    gpus: int,
    strategy: str = "sequential",
    ram_gb: float = config.DEFAULT_RAM_GB,
    gpu_memory_gb: float = config.DEFAULT_GPU_MEMORY_GB,
) -> MemoryResult:

    strategy = strategy.lower().strip()
    gpus = max(int(gpus), 0)

    if gpus == 0:
        # CPU execution: check available system RAM.
        available_gb = ram_gb

    elif strategy == "data_parallel":
        # Each GPU stores a full copy of the model state.
        available_gb = gpu_memory_gb

    elif strategy in ("model_parallel", "hybrid_parallel"):
        # Simplified assumption: model state is evenly distributed.
        available_gb = gpu_memory_gb * gpus

    elif strategy == "distributed":
        # Simplified assumption: each worker has its own GPU memory.
        # This checks aggregate capacity, not actual model placement.
        available_gb = gpu_memory_gb * gpus

    else:
        available_gb = gpu_memory_gb

    available_gb = max(available_gb, 0.001)

    utilization_pct = (memory_required_gb / available_gb) * 100.0
    utilization_pct = min(utilization_pct, 999.0)

    fits = memory_required_gb <= available_gb
    penalty_factor = (
        1.0 if fits else config.MEMORY_PENALTY_FACTOR
    )

    return MemoryResult(
        required_gb=memory_required_gb,
        available_gb=available_gb,
        utilization_pct=utilization_pct,
        fits=fits,
        penalty_factor=penalty_factor,
    )
