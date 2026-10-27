"""
app.py
------
Flask REST API for the HPC-AI Simulator, plus static hosting of the
frontend. Run with:

    python app.py

Then open http://127.0.0.1:5000 in a browser.
"""
import os
from flask import Flask, request, jsonify, send_from_directory

from simulation import config
from simulation.performance import (
    SimulationInput, run_simulation, run_execution_time_comparison,
    run_processor_scaling, run_communication_scaling,
)

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = {
    "dataset_size_gb": (float, 0.001, 100000),
    "model_parameters": (float, 1000, 2_000_000_000_000),
    "batch_size": (int, 1, 65536),
    "epochs": (int, 1, 10000),
    "cpu_cores": (int, 1, 4096),
    "gpus": (int, 0, 4096),
}

OPTIONAL_FIELDS = {
    "gpu_tflops": (float, 0.01, 100000, config.DEFAULT_GPU_TFLOPS),
    "parallelization": (str, None, None, "data_parallel"),
    "network_bandwidth_gbps": (float, 0.001, 1_000_000, config.DEFAULT_NETWORK_BANDWIDTH_GBPS),
    "latency_ms": (float, 0.0, 10000, config.DEFAULT_LATENCY_MS),
    "parallel_fraction": (float, 0.0, 1.0, 0.9),
    "ram_gb": (float, 0.1, 1_000_000, config.DEFAULT_RAM_GB),
    "gpu_memory_gb": (float, 0.1, 1_000_000, config.DEFAULT_GPU_MEMORY_GB),
    "avg_sample_size_mb": (float, 0.0001, 100000, config.DEFAULT_AVG_SAMPLE_SIZE_MB),
    "simd_enabled": (bool, None, None, False),
}


class ValidationError(Exception):
    pass


def validate_and_build(payload: dict) -> SimulationInput:
    if not isinstance(payload, dict):
        raise ValidationError("Request body must be a JSON object.")

    values = {}

    for field, (ftype, lo, hi, *_ ) in REQUIRED_FIELDS.items():
        if field not in payload or payload[field] in (None, ""):
            raise ValidationError(f"Missing required field: '{field}'.")
        try:
            val = ftype(payload[field])
        except (TypeError, ValueError):
            raise ValidationError(f"Field '{field}' must be a number.")
        if lo is not None and not (lo <= val <= hi):
            raise ValidationError(f"Field '{field}' must be between {lo} and {hi}.")
        values[field] = val

    for field, (ftype, lo, hi, default) in OPTIONAL_FIELDS.items():
        raw = payload.get(field, default)
        if raw in (None, ""):
            raw = default
        try:
            val = ftype(raw) if ftype is not bool else bool(raw)
        except (TypeError, ValueError):
            raise ValidationError(f"Field '{field}' has an invalid value.")
        if ftype is not bool and lo is not None and not (lo <= val <= hi):
            raise ValidationError(f"Field '{field}' must be between {lo} and {hi}.")
        values[field] = val

    if values["parallelization"] not in config.PARALLEL_STRATEGIES:
        raise ValidationError(
            f"Field 'parallelization' must be one of {config.PARALLEL_STRATEGIES}."
        )

    return SimulationInput(**values)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    try:
        sim_input = validate_and_build(request.get_json(force=True, silent=True) or {})
        print(
    f"REQUEST DEBUG | GPUs={sim_input.gpus} | "
    f"CPU cores={sim_input.cpu_cores} | "
    f"Strategy={sim_input.parallelization}"
)

        result = run_simulation(sim_input)

        return jsonify(result), 200
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # pragma: no cover - defensive catch-all
        return jsonify({"error": f"Simulation failed: {e}"}), 500


@app.route("/api/graphs/execution-comparison", methods=["POST"])
def api_execution_comparison():
    try:
        sim_input = validate_and_build(request.get_json(force=True, silent=True) or {})
        return jsonify(run_execution_time_comparison(sim_input)), 200
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/graphs/processor-scaling", methods=["POST"])
def api_processor_scaling():
    try:
        payload = request.get_json(force=True, silent=True) or {}
        use_gpu = bool(payload.get("use_gpu", False))
        sim_input = validate_and_build(payload)
        return jsonify(run_processor_scaling(sim_input, use_gpu)), 200
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/graphs/communication-scaling", methods=["POST"])
def api_communication_scaling():
    try:
        sim_input = validate_and_build(request.get_json(force=True, silent=True) or {})
        return jsonify(run_communication_scaling(sim_input)), 200
    except ValidationError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/amdahl", methods=["POST"])
def api_amdahl():
    from simulation.amdahl import amdahl_curve
    payload = request.get_json(force=True, silent=True) or {}
    try:
        p = float(payload.get("parallel_fraction", 0.9))
        p = min(max(p, 0.0), 1.0)
    except (TypeError, ValueError):
        return jsonify({"error": "parallel_fraction must be a number between 0 and 1."}), 400
    return jsonify(amdahl_curve(p)), 200


@app.route("/api/presets", methods=["GET"])
def api_presets():
    return jsonify({
        "scenarios": config.SCENARIO_PRESETS,
        "models": config.MODEL_PRESETS,
        "strategies": config.PARALLEL_STRATEGIES,
        "defaults": {
            "gpu_tflops": config.DEFAULT_GPU_TFLOPS,
            "ram_gb": config.DEFAULT_RAM_GB,
            "gpu_memory_gb": config.DEFAULT_GPU_MEMORY_GB,
            "network_bandwidth_gbps": config.DEFAULT_NETWORK_BANDWIDTH_GBPS,
            "latency_ms": config.DEFAULT_LATENCY_MS,
        },
    }), 200


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
