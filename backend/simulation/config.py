"""
config.py
---------
Every "magic number" used anywhere in the simulation engine lives here,
with a comment explaining the assumption behind it. This is the file a
student should point a professor to when asked "where did that number
come from?".

None of these values are meant to reproduce a specific real GPU/CPU.
They are deliberately simple, round, and documented so the *behaviour*
of the model (how metrics react to changing inputs) is easy to reason
about and explain in a viva. See README.md -> "Simulation Assumptions".
"""

# ---------------------------------------------------------------------------
# Workload / compute-cost model
# ---------------------------------------------------------------------------
# Training a neural network needs roughly 2x FLOPs for the forward pass and
# ~4x FLOPs for the backward pass per parameter per sample. 2 + 4 = 6 is a
# widely used back-of-envelope constant in HPC/ML literature.
FLOPS_PER_PARAM_TRAIN = 6

# Bytes used per parameter. fp32 = 4 bytes. We multiply by 3 in the memory
# model to account for weights + gradients + optimizer state (e.g. Adam).
BYTES_PER_PARAM = 4
MEMORY_STATE_MULTIPLIER = 3  # weights + gradients + optimizer state

# Default average size of a single training sample, used to convert a
# dataset size in GB into an estimated sample count. User configurable.
DEFAULT_AVG_SAMPLE_SIZE_MB = 1.0

# ---------------------------------------------------------------------------
# CPU model
# ---------------------------------------------------------------------------
# Abstract "reference core" performance. Not tied to any real CPU SKU.
BASE_CPU_GFLOPS_PER_CORE = 20.0

# Every extra core adds synchronization / scheduling / memory-bus contention
# overhead, so efficiency decays smoothly instead of scaling linearly.
# efficiency(cores) = 1 / (1 + COEFF * (cores - 1))
CPU_OVERHEAD_COEFF = 0.03
CPU_EFFICIENCY_FLOOR = 0.10

# SIMD / vector processing acceleration factor when enabled (e.g. AVX-style
# wide vector instructions operating on multiple elements per cycle).
VECTOR_ACCEL_FACTOR = 4.0

# ---------------------------------------------------------------------------
# GPU model
# ---------------------------------------------------------------------------
# Efficiency drops when the batch that lands on each device is small
# (the device is under-fed) and as more GPUs are added (sync overhead).
GPU_BASE_EFFICIENCY = 0.9
GPU_BATCH_REFERENCE = 64          # batch size at which a GPU is "well fed"
GPU_OVERHEAD_COEFF = 0.04
GPU_EFFICIENCY_FLOOR = 0.05

# ---------------------------------------------------------------------------
# Memory model
# ---------------------------------------------------------------------------
# If the estimated working-set memory exceeds available capacity, the
# simulator applies a time penalty to represent paging / offloading /
# gradient-checkpointing overhead instead of failing the simulation.
MEMORY_PENALTY_FACTOR = 1.4

# ---------------------------------------------------------------------------
# Communication model (data / model / hybrid / distributed parallelism)
# ---------------------------------------------------------------------------
# Ring all-reduce moves roughly 2x the model size across the network per
# synchronization event (gradient exchange).
ALLREDUCE_FACTOR = 2.0

# Simulating one synchronization per mini-batch is realistic, but for very
# large datasets total_batches can reach the millions, which would make the
# simulation intractable to display meaningfully. We cap the number of
# *simulated* synchronization events and scale accordingly. This is a
# documented simplification — see README "Simulation Assumptions".
MAX_SIMULATED_SYNC_EVENTS = 500

# ---------------------------------------------------------------------------
# Hardware / system defaults (used when the user leaves "advanced" fields
# blank)
# ---------------------------------------------------------------------------
DEFAULT_GPU_TFLOPS = 15.0          # abstract compute capacity per GPU
DEFAULT_RAM_GB = 64.0
DEFAULT_GPU_MEMORY_GB = 16.0
DEFAULT_NETWORK_BANDWIDTH_GBPS = 100.0   # inter-node/inter-GPU link, Gbps
DEFAULT_LATENCY_MS = 0.5

PARALLEL_STRATEGIES = [
    "sequential",
    "cpu_multicore",
    "data_parallel",
    "model_parallel",
    "hybrid_parallel",
    "distributed",
]

MODEL_PRESETS = {
    "small_cnn": {"label": "Small CNN", "parameters": 10_000_000},
    "medium_cnn": {"label": "Medium CNN", "parameters": 100_000_000},
    "large_cnn": {"label": "Large CNN", "parameters": 1_000_000_000},
    "transformer": {"label": "Transformer", "parameters": 10_000_000_000},
    "large_transformer": {"label": "Large Transformer", "parameters": 50_000_000_000},
    "custom": {"label": "Custom Model", "parameters": None},
}

SCENARIO_PRESETS = {
    "small": {
        "name": "Scenario 1 — Small CNN",
        "dataset_size_gb": 1, "model_parameters": 10_000_000,
        "cpu_cores": 4, "gpus": 0, "batch_size": 32, "epochs": 10,
        "parallelization": "cpu_multicore",
    },
    "medium": {
        "name": "Scenario 2 — Medium AI",
        "dataset_size_gb": 10, "model_parameters": 100_000_000,
        "cpu_cores": 8, "gpus": 1, "batch_size": 64, "epochs": 20,
        "parallelization": "data_parallel",
    },
    "large": {
        "name": "Scenario 3 — Large AI",
        "dataset_size_gb": 100, "model_parameters": 10_000_000_000,
        "cpu_cores": 32, "gpus": 4, "batch_size": 128, "epochs": 20,
        "parallelization": "hybrid_parallel",
    },
    "distributed": {
        "name": "Scenario 4 — Distributed AI",
        "dataset_size_gb": 500, "model_parameters": 50_000_000_000,
        "cpu_cores": 64, "gpus": 8, "batch_size": 256, "epochs": 20,
        "parallelization": "distributed",
    },
}
