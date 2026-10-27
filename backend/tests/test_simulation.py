"""
test_simulation.py
-------------------
Covers the test cases described in the project spec (README section
"Testing"). Run with:

    pytest tests/test_simulation.py -v

(run from the backend/ directory so the `simulation` package resolves)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.performance import SimulationInput, run_simulation


def base_input(**overrides) -> SimulationInput:
    defaults = dict(
        dataset_size_gb=10,
        model_parameters=100_000_000,
        batch_size=64,
        epochs=5,
        cpu_cores=1,
        gpus=0,
        parallelization="sequential",
        parallel_fraction=0.9,
    )
    defaults.update(overrides)
    return SimulationInput(**defaults)


def test_1_single_cpu_baseline_speedup_near_one():
    result = run_simulation(base_input(cpu_cores=1, gpus=0, parallelization="sequential"))
    assert abs(result["speedup"] - 1.0) < 0.05


def test_2_increasing_cpu_cores_reduces_time_but_efficiency_drops():
    r_low = run_simulation(base_input(cpu_cores=2, parallelization="cpu_multicore"))
    r_high = run_simulation(base_input(cpu_cores=32, parallelization="cpu_multicore"))
    assert r_high["execution_time"] < r_low["execution_time"]
    assert r_high["speedup"] > r_low["speedup"]
    # Efficiency should eventually decrease due to overhead as cores grow.
    assert r_high["efficiency"] < r_low["efficiency"]


def test_3_adding_gpus_reduces_time_and_increases_throughput():
    r_cpu = run_simulation(base_input(cpu_cores=8, gpus=0, parallelization="cpu_multicore"))
    r_gpu = run_simulation(base_input(cpu_cores=8, gpus=1, parallelization="data_parallel"))
    assert r_gpu["execution_time"] < r_cpu["execution_time"]
    assert r_gpu["throughput_samples_per_sec"] > r_cpu["throughput_samples_per_sec"]


def test_4_more_gpus_show_diminishing_returns_and_more_communication():
    r_4 = run_simulation(base_input(cpu_cores=8, gpus=4, parallelization="data_parallel"))
    r_8 = run_simulation(base_input(cpu_cores=8, gpus=8, parallelization="data_parallel"))
    # Diminishing returns: doubling GPUs should not double the speedup.
    assert r_8["speedup"] < r_4["speedup"] * 2
    assert r_8["communication_overhead_pct"] >= r_4["communication_overhead_pct"]


def test_5_lower_bandwidth_increases_communication_overhead():
    r_fast = run_simulation(base_input(cpu_cores=8, gpus=4, parallelization="data_parallel",
                                        network_bandwidth_gbps=100))
    r_slow = run_simulation(base_input(cpu_cores=8, gpus=4, parallelization="data_parallel",
                                        network_bandwidth_gbps=1))
    assert r_slow["communication_time_sec"] > r_fast["communication_time_sec"]
    assert r_slow["execution_time"] > r_fast["execution_time"]


def test_6_larger_model_increases_time_and_memory():
    r_small = run_simulation(base_input(model_parameters=10_000_000, cpu_cores=8,
                                         parallelization="cpu_multicore"))
    r_large = run_simulation(base_input(model_parameters=1_000_000_000, cpu_cores=8,
                                         parallelization="cpu_multicore"))
    assert r_large["execution_time"] > r_small["execution_time"]
    assert r_large["memory_required_gb"] > r_small["memory_required_gb"]


def test_7_more_epochs_increase_computation_proportionally():
    r_5 = run_simulation(base_input(epochs=5, cpu_cores=8, parallelization="cpu_multicore"))
    r_10 = run_simulation(base_input(epochs=10, cpu_cores=8, parallelization="cpu_multicore"))
    ratio = r_10["workload"]["total_operations_flops"] / r_5["workload"]["total_operations_flops"]
    assert abs(ratio - 2.0) < 0.01


def test_8_batch_size_changes_gpu_efficiency():
    r_small_batch = run_simulation(base_input(cpu_cores=8, gpus=2, batch_size=8,
                                               parallelization="data_parallel"))
    r_large_batch = run_simulation(base_input(cpu_cores=8, gpus=2, batch_size=128,
                                               parallelization="data_parallel"))
    assert r_large_batch["gpu_utilization"] >= r_small_batch["gpu_utilization"]


def test_amdahls_law_bounded_speedup():
    from simulation.amdahl import amdahl_speedup
    # With P=0.5, even infinite processors cannot exceed 1/(1-0.5) = 2x.
    s = amdahl_speedup(0.5, 1_000_000)
    assert s <= 2.01


def test_memory_penalty_applied_when_insufficient():
    from simulation.performance import _core_compute
    sim = base_input(model_parameters=50_000_000_000, cpu_cores=1, gpus=0,
                      parallelization="sequential")
    sim.ram_gb = 1  # force insufficient memory
    t, detail = _core_compute(sim, cores=1, gpus=0, strategy="sequential",
                               simd_enabled=False)
    assert detail["memory"].fits is False
