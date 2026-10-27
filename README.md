# HPC-AI Simulator

### Simulation and Performance Analysis of Deep Learning Workloads on High-Performance Computing Architectures

A web application that simulates how High Performance Computing (HPC) techniques — multicore CPUs, GPU
acceleration, SIMD, data/model parallelism, and distributed computing — affect the performance of training a
deep learning workload. All results come from a transparent, documented mathematical model driven by the
inputs you choose. Nothing is randomized to "look realistic," and no real neural network is trained.

---

## 1. Problem Statement

Large-scale AI workloads require significant computational resources. HPC techniques such as parallel
processing, GPU acceleration, and distributed computing can reduce execution time, but understanding the
relationship between workload, hardware, parallelism, and performance can be difficult without hands-on
experimentation. This project provides a simulation environment for studying those relationships: change an
input, and watch the effect propagate through workload size, parallelization, hardware throughput, memory
behavior, communication overhead, execution time, speedup, and efficiency.

## 2. Objectives

1. Simulate AI (deep learning) training workloads from configurable parameters.
2. Demonstrate core HPC architecture concepts (CPU, GPU, cluster/distributed).
3. Compare CPU-only vs. GPU-accelerated processing.
4. Demonstrate parallel processing and multicore scaling.
5. Demonstrate data parallelism and model parallelism.
6. Model communication overhead in multi-device/multi-node training.
7. Calculate speedup and parallel efficiency.
8. Demonstrate scalability and diminishing returns.
9. Demonstrate Amdahl's Law.
10. Provide visual, interactive performance analysis (charts, comparison mode, experiment sweeps).

## 3. Technologies

```text
Backend:   Python 3, Flask (REST API)
Frontend:  HTML5, CSS3, vanilla JavaScript
Charts:    Chart.js (vendored locally, no CDN dependency)
Testing:   pytest
```

## 4. Project Structure

```text
hpc-ai-simulator/
├── backend/
│   ├── app.py                     Flask app: routes, input validation, static hosting
│   ├── requirements.txt
│   ├── tests/
│   │   └── test_simulation.py     Automated tests (see section 9)
│   └── simulation/                The performance model, one concern per file
│       ├── config.py              Every constant/assumption, documented
│       ├── workload.py            Workload Analyzer
│       ├── cpu_model.py           CPU Performance Model
│       ├── gpu_model.py           GPU Performance Model
│       ├── memory_model.py        Memory hierarchy / capacity model
│       ├── communication.py       Communication overhead model
│       ├── parallelism.py         Parallelization Engine (strategy → resources)
│       ├── amdahl.py              Amdahl's Law
│       └── performance.py         Orchestrator: wires all of the above together
└── frontend/
    ├── index.html                 Single-page app shell (Workload / Architecture /
    │                               Simulation / Results / Experiments / About)
    ├── css/style.css
    └── js/
        ├── app.js                 Navigation, form handling, API calls, rendering
        ├── charts.js              Chart.js chart construction/update
        ├── architecture.js        Static architecture/SIMD/memory/parallelism diagrams
        ├── help-content.js        "What does this mean?" explanations
        └── vendor/chart.umd.js    Chart.js, vendored so the app works fully offline
```

## 5. Architecture / Data Flow

```text
AI Workload (dataset, model, batch size, epochs)
        │
        ▼
Workload Analyzer          → total samples, total batches, total FLOPs, memory footprint
        │
        ▼
Parallelization Engine     → resolves strategy into concrete CPU cores / GPUs / devices used
        │
   ┌────┼────────────┐
   ▼    ▼             ▼
  CPU   GPU        (Distributed = CPU+GPU treated as a multi-node cluster)
   │    │
   └────┼──────────────┘
        ▼
Memory Model               → does the working set fit? apply penalty if not
        ▼
Communication Model        → sync time from model size, bandwidth, latency, device count
        ▼
Performance Model          → execution time, speedup, efficiency, throughput, utilization
        ▼
Results Dashboard + Charts + Amdahl's Law + Comparison Mode + Experiment Mode
```

Every simulation run re-executes this entire pipeline — there is no cached/mocked output. Changing GPUs from
1 → 4, for example, changes the parallelization plan → GPU compute throughput → memory footprint per device →
communication volume → execution time → speedup → efficiency, in that order.

## 6. Simulation Formulas

All constants referenced below live in `backend/simulation/config.py`, each with a comment explaining the
assumption.

**Workload Analyzer** (`workload.py`)
```text
Total Samples       = Dataset Size (MB) / Average Sample Size (MB)
Batches per Epoch   = Total Samples / Batch Size
Total Batches       = Batches per Epoch × Epochs
Ops per Sample      = Model Parameters × 6      (2× forward + 4× backward FLOPs/param)
Total Operations    = Ops per Sample × Total Samples × Epochs
Memory Required     = (Model Parameters × 4 bytes × 3 [weights+grads+optimizer state])
                        + (Batch Size × Avg Sample Size)
```

**CPU Model** (`cpu_model.py`)
```text
CPU Efficiency          = 1 / (1 + 0.03 × (cores − 1))         [floor 10%]
Effective CPU GFLOPS     = Base GFLOPS/core × cores × efficiency × (4× if SIMD enabled)
```

**GPU Model** (`gpu_model.py`)
```text
Batch Factor       = min(1, batch_size / 64)                    [64 = "well-fed" reference batch]
Scaling Penalty     = 1 + 0.04 × (GPUs − 1)
GPU Efficiency      = (0.9 × Batch Factor) / Scaling Penalty     [floor 5%]
Effective GPU GFLOPS = GPU TFLOPS × 1000 × GPUs × efficiency
```

**Memory Model** (`memory_model.py`)
```text
Available Memory   = GPU Memory/device × GPUs   (or RAM, if no GPUs)
If Required > Available: apply a 1.4× time penalty (paging / offload simulation)
```

**Communication Model** (`communication.py`) — applies to data/model/hybrid/distributed strategies with >1 device
```text
Data per sync event   = Model Parameters × 4 bytes × 2    (ring all-reduce ≈ 2× model size)
Communication Time    = (sync events × data/event ÷ bandwidth) + (sync events × latency × device-count factor)
```
*Documented simplification:* one amortized synchronization event per epoch, rather than per mini-batch, to
keep the numbers meaningful for datasets with millions of batches (see `communication.py` docstring).

**Amdahl's Law** (`amdahl.py`)
```text
Speedup(N) = 1 / ((1 − P) + (P / N))
```

**Performance Model** (`performance.py`, orchestrator)
```text
Execution Time   = Compute Time (workload ÷ effective GFLOPS, × memory penalty) + Communication Time
Speedup          = Baseline (1 CPU core, sequential) Execution Time / Execution Time
Efficiency       = (Speedup / Processing Units) × 100
Throughput       = Total Samples Processed / Execution Time
```

## 7. API Design

Base URL: `http://127.0.0.1:5000/api`

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/simulate` | Run the full model for one configuration, returns all metrics |
| POST | `/graphs/execution-comparison` | Sequential / CPU / GPU / Multi-GPU / Distributed execution times |
| POST | `/graphs/processor-scaling` | Speedup & efficiency vs. processor count (`use_gpu: true/false`) |
| POST | `/graphs/communication-scaling` | Communication time/overhead vs. device count |
| POST | `/amdahl` | Amdahl's Law curve for a given parallelizable fraction |
| GET  | `/presets` | Scenario presets, model presets, strategy list, defaults |

**Request body for `/simulate`:**
```json
{
  "dataset_size_gb": 10,
  "model_parameters": 100000000,
  "cpu_cores": 8,
  "gpus": 2,
  "batch_size": 64,
  "epochs": 20,
  "parallelization": "data_parallel",
  "network_bandwidth_gbps": 100,
  "parallel_fraction": 0.9
}
```

**Response (abridged):**
```json
{
  "execution_time": 211.3492,
  "speedup": 29.07,
  "efficiency": 1453.52,
  "throughput_samples_per_sec": 969.01,
  "cpu_utilization": 82.6,
  "gpu_utilization": 86.5,
  "memory_utilization": 3.9,
  "communication_overhead_pct": 97.77,
  "amdahl_curve": [ { "processors": 1, "speedup": 1.0 }, ... ],
  "parallel_plan": { "strategy": "data_parallel", "effective_cores": 8, "effective_gpus": 2, ... }
}
```

All endpoints validate input types and ranges and return `400` with a descriptive `error` message on invalid
input, or `500` on an unexpected internal failure.

## 8. Frontend Pages

- **Workload** — configuration form (AI workload, hardware, parallelization strategy, advanced system
  parameters) plus scenario presets and the **RUN SIMULATION** button.
- **Architecture** — the conceptual pipeline diagram, an SIMD scalar-vs-vector demo, the memory hierarchy,
  and data/model parallelism illustrations.
- **Simulation** — an animated, staged trace of the pipeline (Workload Analyzer → Parallelization →
  Compute → Memory → Communication → Performance Model) with the live numeric result of each stage.
- **Results** — KPI cards, the parallelization plan, six Chart.js graphs (execution comparison, speedup vs.
  processors, efficiency vs. processors, CPU vs. GPU, interactive Amdahl's Law, communication overhead vs.
  devices), and Comparison Mode (save/compare multiple runs in a table).
- **Experiments** — four preset sweeps (CPU cores, GPU count, batch size, Amdahl's parallelizable fraction)
  that re-run `/simulate` across a range of values and plot the trend.
- **About** — plain-language explanations of Deep Learning and HPC, and the full Simulation Assumptions
  list (section 10 below).

Every metric has a "?" help button opening a plain-language explanation (Speedup, Efficiency, Communication
Overhead, Amdahl's Law, Data/Model Parallelism, etc.).

## 9. Testing

Automated tests live in `backend/tests/test_simulation.py` (run with `pytest`, from the `backend/` directory).
They cover the exact scenarios from the project brief:

| # | Test | Expectation |
|---|---|---|
| 1 | 1 CPU core, sequential | Speedup ≈ 1× (baseline) |
| 2 | Increase CPU cores (2 → 32) | Execution time falls, speedup rises, efficiency eventually falls |
| 3 | Add a GPU (CPU-only vs. CPU+1 GPU) | Execution time falls, throughput rises |
| 4 | Increase GPUs significantly (4 → 8) | Diminishing speedup returns; communication overhead rises |
| 5 | Reduce network bandwidth (100 → 1 Gbps) | Communication time and execution time increase |
| 6 | Increase model size (10M → 1B params) | Execution time and memory requirement increase |
| 7 | Increase epochs (5 → 10) | Total FLOPs scale proportionally (≈2×) |
| 8 | Increase batch size (8 → 128) | GPU utilization responds per the batch/efficiency model |
| — | Amdahl's Law | Speedup is bounded by 1 / (1 − P) even with huge processor counts |
| — | Memory penalty | Insufficient memory triggers `fits = false` and a time penalty |

Run them:
```bash
cd backend
pytest tests/ -v
```
All 10 tests pass against the current model.

## 10. Simulation Assumptions (read this before presenting the results)

- Execution time is an **estimate** from a simplified mathematical model — not a measurement of real hardware.
- CPU and GPU performance figures (GFLOPS/TFLOPS) are abstract "reference" values, not benchmarks of any
  specific real chip, and the app works with **no physical GPU present**.
- Communication overhead is estimated from model size, network bandwidth, and latency using a simplified
  ring all-reduce model, amortized to one synchronization event per epoch (see `communication.py`).
- Memory behavior is simplified to a single working-set estimate (weights + gradients + optimizer state +
  one batch), compared against a single "available memory" figure.
- CPU / GPU / memory utilization percentages are **simulated**, not readings from the machine running the app.
- Because the model's baseline is a single CPU core, configurations that add GPUs can show speedup and
  "efficiency" figures well above what per-core CPU scaling alone would ever reach (efficiency here is
  literally `speedup ÷ processing units`, so a GPU that is hundreds of times faster than one CPU core will
  push efficiency past 100% when only a handful of GPUs are counted as "processing units"). This is expected
  behavior of the documented formula, not a bug — it illustrates why "efficiency" is normally compared
  *within* one hardware class (e.g., CPU cores vs. CPU cores) rather than across CPU and GPU baselines.
- Scenario presets are illustrative teaching examples, not calibrated real-world training benchmarks.
- This tool is intended for **educational comparison of HPC concepts**, not production capacity planning.

## 11. How to Run

Requires Python 3.9+.

```bash
# 1. Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Run the Flask app (also serves the frontend)
cd backend
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. No build step, no internet connection required — Chart.js
is bundled locally in `frontend/js/vendor/`.

## 12. What This Project Is Not

This is a college HPC/performance-modeling project, not a production system. It intentionally does **not**
include: a real distributed training framework, real CUDA/MPI execution, an actual neural-network training
loop, or a production cloud capacity planner. The goal is to make the relationship

```text
Input → Workload → Parallelism → Hardware Model → Communication → Performance → Results
```

transparent and explainable, end to end.
