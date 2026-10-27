"""
gpu_model.py
------------
GPU PERFORMANCE MODEL

GPU Performance = GPU Compute Capacity x Number of GPUs x GPU Efficiency

GPU efficiency depends on:
  - batch size (small batches under-feed the device)
  - number of GPUs (more devices -> more synchronization overhead)
Scaling is intentionally sub-linear, never perfect.
"""
from dataclasses import dataclass
from . import config


@dataclass
class GPUResult:
    gpus: int
    gpu_tflops: float
    efficiency: float
    effective_gflops: float


def compute_gpu_performance(gpus: int, gpu_tflops: float, batch_size: int) -> GPUResult:
    gpus = max(int(gpus), 0)
    if gpus == 0:
        return GPUResult(gpus=0, gpu_tflops=gpu_tflops, efficiency=0.0, effective_gflops=0.0)

    # Small batches under-utilize a GPU's parallel lanes.
    batch_factor = min(1.0, batch_size / config.GPU_BATCH_REFERENCE)

    # More GPUs => more gradient-sync overhead reduces efficiency.
    scaling_penalty = 1.0 + config.GPU_OVERHEAD_COEFF * (gpus - 1)

    efficiency = (config.GPU_BASE_EFFICIENCY * batch_factor) / scaling_penalty
    efficiency = max(min(efficiency, config.GPU_BASE_EFFICIENCY), config.GPU_EFFICIENCY_FLOOR)

    effective_gflops = gpu_tflops * 1000.0 * gpus * efficiency

    return GPUResult(
        gpus=gpus,
        gpu_tflops=gpu_tflops,
        efficiency=efficiency,
        effective_gflops=effective_gflops,
    )
