
"""
performance.py
---------------
PERFORMANCE MODEL (orchestrator)

Combines workload analysis, CPU/GPU performance, memory,
communication, parallelization, and Amdahl's Law into
simulation results.
"""

from dataclasses import dataclass, asdict

from . import config
from .workload import analyze_workload
from .cpu_model import compute_cpu_performance
from .gpu_model import compute_gpu_performance
from .memory_model import evaluate_memory
from .communication import evaluate_communication
from .parallelism import resolve_strategy
from .amdahl import amdahl_speedup, amdahl_curve


@dataclass
class SimulationInput:
    dataset_size_gb: float
    model_parameters: int
    batch_size: int
    epochs: int
    cpu_cores: int
    gpus: int
    gpu_tflops: float = config.DEFAULT_GPU_TFLOPS
    parallelization: str = "data_parallel"
    network_bandwidth_gbps: float = config.DEFAULT_NETWORK_BANDWIDTH_GBPS
    latency_ms: float = config.DEFAULT_LATENCY_MS
    parallel_fraction: float = 0.9
    ram_gb: float = config.DEFAULT_RAM_GB
    gpu_memory_gb: float = config.DEFAULT_GPU_MEMORY_GB
    avg_sample_size_mb: float = config.DEFAULT_AVG_SAMPLE_SIZE_MB
    simd_enabled: bool = False


def _core_compute(
    sim: SimulationInput,
    cores: int,
    gpus: int,
    strategy: str,
    simd_enabled: bool,
):
    """Estimate execution time for a given resource configuration."""

    workload = analyze_workload(
        sim.dataset_size_gb,
        sim.model_parameters,
        sim.batch_size,
        sim.epochs,
        sim.avg_sample_size_mb,
    )

    cpu = compute_cpu_performance(cores, simd_enabled)

    gpu = compute_gpu_performance(
        gpus,
        sim.gpu_tflops,
        sim.batch_size,
    )

    # Combined estimated CPU and GPU compute throughput.
    total_gflops = cpu.effective_gflops + gpu.effective_gflops
    total_gflops = max(total_gflops, 0.001)

    compute_time_sec = workload.total_operations / (total_gflops * 1e9)

    memory = evaluate_memory(
    memory_required_gb=workload.memory_required_gb,
    gpus=gpus,
    strategy=strategy,
    ram_gb=sim.ram_gb,
    gpu_memory_gb=sim.gpu_memory_gb,
)

    # Apply the memory performance penalty.
    compute_time_sec *= memory.penalty_factor

    plan = resolve_strategy(strategy, cores, gpus)

    if plan.communication_applies:
        comm = evaluate_communication(
            strategy,
            plan.devices,
            sim.epochs,
            sim.model_parameters,
            sim.network_bandwidth_gbps,
            sim.latency_ms,
        )
    else:
        comm = evaluate_communication(
            "sequential",
            1,
            workload.total_batches,
            sim.model_parameters,
        )

    execution_time_sec = (
        compute_time_sec + comm.communication_time_sec
    )

    return execution_time_sec, {
        "workload": workload,
        "cpu": cpu,
        "gpu": gpu,
        "memory": memory,
        "communication": comm,
        "plan": plan,
        "compute_time_sec": compute_time_sec,
    }


def run_simulation(sim: SimulationInput) -> dict:
    """Run the main simulation and return dashboard metrics."""

    # Baseline: one CPU core, no GPU, no SIMD.
    baseline_time, _ = _core_compute(
        sim,
        cores=1,
        gpus=0,
        strategy="sequential",
        simd_enabled=False,
    )

    plan = resolve_strategy(
        sim.parallelization,
        sim.cpu_cores,
        sim.gpus,
    )

    exec_time, detail = _core_compute(
        sim,
        cores=plan.effective_cores,
        gpus=plan.effective_gpus,
        strategy=sim.parallelization,
        simd_enabled=sim.simd_enabled,
    )

    exec_time = max(exec_time, 1e-9)
    baseline_time = max(baseline_time, 1e-9)

    workload = detail["workload"]
    cpu = detail["cpu"]
    gpu = detail["gpu"]
    memory = detail["memory"]
    comm = detail["communication"]



    # Speedup is explicitly relative to the single-CPU-core baseline.
    speedup = baseline_time / exec_time

    # Calculate efficiency using a matching resource baseline.
    # GPU configurations are compared against one GPU with the
    # same CPU-core count and parallelization strategy.
    if plan.effective_gpus > 0:
        # Compare one GPU against the configured GPU count.
        gpu_baseline_time, _ = _core_compute(
            sim,
            cores=plan.effective_cores,
            gpus=1,
            strategy=sim.parallelization,
            simd_enabled=sim.simd_enabled,
        )

        gpu_baseline_time = max(gpu_baseline_time, 1e-9)

        gpu_speedup = gpu_baseline_time / exec_time
        efficiency = (
            gpu_speedup / max(plan.effective_gpus, 1)
        ) * 100.0

    else:
        # CPU-only efficiency relative to one CPU core.
        cpu_speedup = baseline_time / exec_time
        efficiency = (
            cpu_speedup / max(plan.effective_cores, 1)
        ) * 100.0

    efficiency = min(max(efficiency, 0.0), 100.0)

    total_samples_processed = workload.total_samples * sim.epochs

    throughput_samples = total_samples_processed / exec_time
    throughput_batches = throughput_samples / max(sim.batch_size, 1)

    communication_overhead_pct = (
        (comm.communication_time_sec / exec_time) * 100.0
        if exec_time > 0
        else 0.0
    )

    cpu_utilization = round(cpu.efficiency * 100.0, 1)

    gpu_utilization = (
        round(gpu.efficiency * 100.0, 1)
        if plan.effective_gpus > 0
        else 0.0
    )

    active_units = (
        plan.effective_cores
        if plan.effective_gpus == 0
        else plan.effective_gpus
    )

    energy_index = round(exec_time * active_units * 0.01, 2)

    amdahl_theoretical_speedup = amdahl_speedup(
        sim.parallel_fraction,
        max(plan.devices, 1),
    )

    return {
        "inputs": asdict(sim),
        "baseline_execution_time": round(baseline_time, 4),
        "execution_time": round(exec_time, 4),
        "speedup": round(speedup, 3),
        "efficiency": round(efficiency, 2),
        "throughput_samples_per_sec": round(throughput_samples, 2),
        "throughput_batches_per_sec": round(throughput_batches, 4),
        "cpu_utilization": cpu_utilization,
        "gpu_utilization": gpu_utilization,
        "memory_utilization": round(memory.utilization_pct, 1),
        "memory_required_gb": round(memory.required_gb, 3),
        "memory_available_gb": round(memory.available_gb, 3),
        "memory_fits": memory.fits,
        "communication_time_sec": round(
            comm.communication_time_sec, 4
        ),
        "communication_overhead_pct": round(
            communication_overhead_pct, 2
        ),
        "compute_time_sec": round(
            detail["compute_time_sec"], 4
        ),
        "energy_index": energy_index,
        "amdahl_theoretical_speedup": round(
            amdahl_theoretical_speedup, 3
        ),
        "amdahl_curve": amdahl_curve(sim.parallel_fraction),
        "workload": {
            "total_samples": round(workload.total_samples, 1),
            "batches_per_epoch": round(
                workload.batches_per_epoch, 2
            ),
            "total_batches": round(workload.total_batches, 2),
            "total_operations_flops": workload.total_operations,
        },
        "parallel_plan": {
            "strategy": plan.strategy,
            "effective_cores": plan.effective_cores,
            "effective_gpus": plan.effective_gpus,
            "devices": plan.devices,
            "communication_applies": plan.communication_applies,
            "description": plan.description,
        },
    }



def run_execution_time_comparison(sim: SimulationInput) -> list:
    """Compare execution times across CPU and GPU configurations."""

    variants = [
        ("Sequential", 1, 0, "sequential"),
        ("CPU Multicore", sim.cpu_cores, 0, "cpu_multicore"),
        ("GPU (1 device)", sim.cpu_cores, 1, "data_parallel"),
        ("Multi-GPU", sim.cpu_cores, max(sim.gpus, 2), "data_parallel"),
        ("Distributed", sim.cpu_cores, max(sim.gpus, 4), "distributed"),
    ]

    results = []

    for label, cores, gpus, strategy in variants:
        t, _ = _core_compute(
            sim,
            cores=cores,
            gpus=gpus,
            strategy=strategy,
            simd_enabled=sim.simd_enabled,
        )

        results.append({
            "label": label,
            "execution_time": round(t, 4),
        })

    return results



def run_processor_scaling(
    sim: SimulationInput,
    use_gpu: bool,
    processor_points=None,
) -> list:
    """Return speedup and efficiency for CPU or GPU scaling."""

    if processor_points is None:
        processor_points = [1, 2, 4, 8, 16, 32, 64]

    if use_gpu:
        # GPU scaling baseline: one GPU and the same CPU configuration.
        baseline_time, _ = _core_compute(
            sim,
            cores=sim.cpu_cores,
            gpus=1,
            strategy="data_parallel",
            simd_enabled=sim.simd_enabled,
        )
    else:
        # CPU scaling baseline: one CPU core.
        baseline_time, _ = _core_compute(
            sim,
            cores=1,
            gpus=0,
            strategy="sequential",
            simd_enabled=False,
        )

    baseline_time = max(baseline_time, 1e-9)
    results = []

    for n in processor_points:
        n = max(int(n), 1)

        if use_gpu:
            t, _ = _core_compute(
                sim,
                cores=sim.cpu_cores,
                gpus=n,
                strategy="data_parallel",
                simd_enabled=sim.simd_enabled,
            )
        else:
            t, _ = _core_compute(
                sim,
                cores=n,
                gpus=0,
                strategy="cpu_multicore",
                simd_enabled=sim.simd_enabled,
            )

        t = max(t, 1e-9)

        speedup = baseline_time / t
        efficiency = (speedup / n) * 100.0

        results.append({
            "processors": n,
            "execution_time": round(t, 4),
            "speedup": round(speedup, 3),
            "efficiency": round(efficiency, 2),
        })

    return results


def run_communication_scaling(
    sim: SimulationInput,
    device_points=None,
) -> list:
    """Estimate communication overhead at different device counts."""

    if device_points is None:
        device_points = [1, 2, 4, 8, 16, 32]

    results = []

    for n in device_points:
        strategy = (
            sim.parallelization
            if sim.parallelization in (
                "data_parallel",
                "model_parallel",
                "hybrid_parallel",
                "distributed",
            )
            else "data_parallel"
        )

        t, detail = _core_compute(
            sim,
            cores=sim.cpu_cores,
            gpus=n,
            strategy=strategy,
            simd_enabled=sim.simd_enabled,
        )

        comm_time = detail["communication"].communication_time_sec

        pct = (
            (comm_time / t) * 100.0
            if t > 0
            else 0.0
        )

        results.append({
            "devices": n,
            "communication_time_sec": round(comm_time, 4),
            "communication_overhead_pct": round(pct, 2),
        })

    return results
