/**
 * app.js
 * -------
 * Wires the UI together: page navigation, the configuration form, calls to
 * the Flask REST API, the simulation animation, and rendering of results,
 * comparison table, and experiments. No mock/random data — every number
 * shown comes straight from an /api/* response.
 */

const API_BASE = '/api';
let lastResult = null;
let comparisonRows = [];
let presetsCache = null;

// ---------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------
function goToPage(pageId) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById(`page-${pageId}`).classList.add('active');
  document.querySelector(`.nav-item[data-page="${pageId}"]`).classList.add('active');
}

document.querySelectorAll('.nav-item').forEach(btn => {
  btn.addEventListener('click', () => goToPage(btn.dataset.page));
});

// ---------------------------------------------------------------------
// Theme toggle
// ---------------------------------------------------------------------
document.getElementById('theme-toggle').addEventListener('click', () => {
  const html = document.documentElement;
  const current = html.getAttribute('data-theme');
  html.setAttribute('data-theme', current === 'light' ? 'dark' : 'light');
});

// ---------------------------------------------------------------------
// Form helpers
// ---------------------------------------------------------------------
const MODEL_PARAM_MAP = {
  small_cnn: 10_000_000,
  medium_cnn: 100_000_000,
  large_cnn: 1_000_000_000,
  transformer: 10_000_000_000,
  large_transformer: 50_000_000_000,
};

const GPU_TFLOPS_MAP = { none: 0, generic: 10, highend: 15, accelerator: 40 };

document.getElementById('model-type').addEventListener('change', (e) => {
  const val = e.target.value;
  if (val !== 'custom' && MODEL_PARAM_MAP[val]) {
    document.getElementById('model_parameters').value = MODEL_PARAM_MAP[val];
  }
});

document.getElementById('gpu_type').addEventListener('change', (e) => {
  const tflops = GPU_TFLOPS_MAP[e.target.value];
  if (tflops !== undefined) document.getElementById('gpu_tflops').value = tflops;
  if (e.target.value === 'none') document.getElementById('gpus').value = '0';
});

const pfSlider = document.getElementById('parallel_fraction');
const pfDisplay = document.getElementById('parallel_fraction_display');
pfSlider.addEventListener('input', () => {
  const p = parseFloat(pfSlider.value);
  pfDisplay.textContent = `P = ${p.toFixed(2)} (${Math.round(p * 100)}% parallelizable)`;
});

function readFormValues() {
  return {
    dataset_size_gb: parseFloat(document.getElementById('dataset_size_gb').value),
    model_parameters: parseFloat(document.getElementById('model_parameters').value),
    batch_size: parseInt(document.getElementById('batch_size').value, 10),
    epochs: parseInt(document.getElementById('epochs').value, 10),
    cpu_cores: parseInt(document.getElementById('cpu_cores').value, 10),
    gpus: parseInt(document.getElementById('gpus').value, 10),
    gpu_tflops: parseFloat(document.getElementById('gpu_tflops').value),
    parallelization: document.getElementById('parallelization').value,
    network_bandwidth_gbps: parseFloat(document.getElementById('network_bandwidth_gbps').value),
    latency_ms: parseFloat(document.getElementById('latency_ms').value),
    parallel_fraction: parseFloat(document.getElementById('parallel_fraction').value),
    ram_gb: parseFloat(document.getElementById('ram_gb').value),
    gpu_memory_gb: parseFloat(document.getElementById('gpu_memory_gb').value),
    avg_sample_size_mb: parseFloat(document.getElementById('avg_sample_size_mb').value),
    simd_enabled: document.getElementById('simd_enabled').checked,
  };
}

function applyPreset(key) {
  const presets = {
    small:  { dataset_size_gb: 1, model_parameters: 10_000_000, cpu_cores: 4, gpus: 0, batch_size: 32, epochs: 10, parallelization: 'cpu_multicore' },
    medium: { dataset_size_gb: 10, model_parameters: 100_000_000, cpu_cores: 8, gpus: 1, batch_size: 64, epochs: 20, parallelization: 'data_parallel' },
    large:  { dataset_size_gb: 100, model_parameters: 10_000_000_000, cpu_cores: 32, gpus: 4, batch_size: 128, epochs: 20, parallelization: 'hybrid_parallel' },
    distributed: { dataset_size_gb: 500, model_parameters: 50_000_000_000, cpu_cores: 64, gpus: 8, batch_size: 256, epochs: 20, parallelization: 'distributed' },
  };
  const p = presets[key];
  if (!p) return;
  document.getElementById('dataset_size_gb').value = p.dataset_size_gb;
  document.getElementById('model_parameters').value = p.model_parameters;
  const modelSelect = document.getElementById('model-type');

const matchingModel = Object.entries(MODEL_PARAM_MAP).find(
  ([key, parameters]) => parameters === p.model_parameters
);

if (matchingModel) {
  modelSelect.value = matchingModel[0];
} else {
  modelSelect.value = 'custom';
}
  document.getElementById('cpu_cores').value = String(p.cpu_cores);
  document.getElementById('gpus').value = String(p.gpus);
  document.getElementById('batch_size').value = String(p.batch_size);
  document.getElementById('epochs').value = p.epochs;
  document.getElementById('parallelization').value = p.parallelization;

  document.querySelectorAll('.preset-bar .chip[data-preset]').forEach(c => c.classList.remove('active'));
  document.querySelector(`.chip[data-preset="${key}"]`).classList.add('active');
}

document.querySelectorAll('.chip[data-preset]').forEach(chip => {
  chip.addEventListener('click', () => applyPreset(chip.dataset.preset));
});

// ---------------------------------------------------------------------
// API helper
// ---------------------------------------------------------------------
async function apiPost(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `Request to ${path} failed`);
  return data;
}

// ---------------------------------------------------------------------
// Simulation animation (pipeline trace)
// ---------------------------------------------------------------------
const PIPELINE_STAGES = [
  'Workload Analyzer', 'Parallelization Engine', 'CPU / GPU Compute',
  'Memory Model', 'Communication', 'Performance Model',
];

function renderSimStages(activeIndex, doneUpTo) {
  const el = document.getElementById('sim-animation');
  el.innerHTML = `<div class="sim-stage-row">${PIPELINE_STAGES.map((s, i) => {
    let cls = '';
    if (i < doneUpTo) cls = 'done';
    if (i === activeIndex) cls = 'active';
    return `<div class="sim-stage ${cls}">${s}</div>`;
  }).join('<span class="arch-arrow">&#8594;</span>')}</div>`;
}

function logLine(msg) {
  const log = document.getElementById('sim-log');
  const card = document.getElementById('sim-log-card');
  card.style.display = 'block';
  const line = document.createElement('div');
  line.innerHTML = msg;
  log.appendChild(line);
  log.scrollTop = log.scrollHeight;
}

async function playSimulationAnimation(config, result) {
  goToPage('simulation');
  document.getElementById('sim-log').innerHTML = '';
  document.getElementById('sim-log-card').style.display = 'none';
  const delay = (ms) => new Promise(r => setTimeout(r, ms));

  const messages = [
    `Workload Analyzer &rarr; ${Math.round(result.workload.total_samples).toLocaleString()} samples, ${result.workload.total_batches.toLocaleString()} total batches, ${result.workload.total_operations_flops.toExponential(2)} FLOPs`,
    `Parallelization Engine &rarr; strategy "${result.parallel_plan.strategy}": ${result.parallel_plan.effective_cores} cores, ${result.parallel_plan.effective_gpus} GPUs, ${result.parallel_plan.devices} processing units`,
    `CPU / GPU Compute &rarr; compute time <span class="log-ok">${result.compute_time_sec}s</span>`,
    `Memory Model &rarr; ${result.memory_required_gb} GB required / ${result.memory_available_gb} GB available (${result.memory_fits ? 'fits' : 'penalty applied'})`,
    `Communication &rarr; ${result.communication_time_sec}s (${result.communication_overhead_pct}% of total)`,
    `Performance Model &rarr; execution time <span class="log-ok">${result.execution_time}s</span>, speedup <span class="log-ok">${result.speedup}x</span>, efficiency <span class="log-ok">${result.efficiency}%</span>`,
  ];

  for (let i = 0; i < PIPELINE_STAGES.length; i++) {
    renderSimStages(i, i);
    await delay(280);
    logLine(messages[i]);
  }
  renderSimStages(-1, PIPELINE_STAGES.length);
}

// ---------------------------------------------------------------------
// Results rendering
// ---------------------------------------------------------------------
function renderResults(result) {
  document.getElementById('no-results').style.display = 'none';
  document.getElementById('results-content').style.display = 'block';

  document.getElementById('kpi-exec-time').textContent = `${result.execution_time}s`;
  document.getElementById('kpi-speedup').textContent = `${result.speedup}x`;
  document.getElementById('kpi-efficiency').textContent = `${result.efficiency}%`;
  document.getElementById('kpi-throughput').textContent = result.throughput_samples_per_sec.toLocaleString();

  document.getElementById('kpi-cpu-util').textContent = `${result.cpu_utilization}%`;
  document.getElementById('kpi-gpu-util').textContent = `${result.gpu_utilization}%`;
  document.getElementById('kpi-mem-util').textContent = `${result.memory_utilization}%`;
  document.getElementById('kpi-comm').textContent = `${result.communication_overhead_pct}%`;
  document.getElementById('kpi-energy').textContent = result.energy_index;

  const memWarn = document.getElementById('memory-warning-card');
  if (!result.memory_fits) {
    memWarn.style.display = 'block';
    document.getElementById('mem-required').textContent = result.memory_required_gb;
    document.getElementById('mem-available').textContent = result.memory_available_gb;
  } else {
    memWarn.style.display = 'none';
  }

  document.getElementById('plan-description').textContent = result.parallel_plan.description;
  document.getElementById('plan-grid').innerHTML = `
    <div class="plan-item"><span class="plan-key">Strategy</span><span class="plan-val">${result.parallel_plan.strategy}</span></div>
    <div class="plan-item"><span class="plan-key">Effective CPU Cores</span><span class="plan-val">${result.parallel_plan.effective_cores}</span></div>
    <div class="plan-item"><span class="plan-key">Effective GPUs</span><span class="plan-val">${result.parallel_plan.effective_gpus}</span></div>
    <div class="plan-item"><span class="plan-key">Processing Units</span><span class="plan-val">${result.parallel_plan.devices}</span></div>
    <div class="plan-item"><span class="plan-key">Communication Applies</span><span class="plan-val">${result.parallel_plan.communication_applies}</span></div>
  `;
}

function addComparisonRow(config, result) {
  const row = {
    label: `Run ${comparisonRows.length + 1} (${config.parallelization})`,
    cpu: config.cpu_cores,
    gpus: config.gpus,
    strategy: config.parallelization,
    time: result.execution_time,
    speedup: result.speedup,
    efficiency: result.efficiency,
    throughput: result.throughput_samples_per_sec,
  };
  comparisonRows.push(row);
  renderComparisonTable();
}

function renderComparisonTable() {
  const tbody = document.getElementById('comparison-tbody');
  tbody.innerHTML = comparisonRows.map((r, i) => `
    <tr>
      <td>${r.label}</td><td>${r.cpu}</td><td>${r.gpus}</td><td>${r.strategy}</td>
      <td>${r.time}</td><td>${r.speedup}x</td><td>${r.efficiency}%</td><td>${r.throughput.toLocaleString()}</td>
      <td class="row-remove"><button data-idx="${i}">remove</button></td>
    </tr>
  `).join('');
  tbody.querySelectorAll('button[data-idx]').forEach(btn => {
    btn.addEventListener('click', () => {
      comparisonRows.splice(parseInt(btn.dataset.idx, 10), 1);
      renderComparisonTable();
    });
  });
}

document.getElementById('save-experiment-btn').addEventListener('click', () => {
  if (!lastResult) return;
  addComparisonRow(lastResult.inputs, lastResult);
});

// ---------------------------------------------------------------------
// Run simulation
// ---------------------------------------------------------------------
async function runFullSimulation() {
  const btn = document.getElementById('run-simulation-btn');
  const errEl = document.getElementById('form-error');
  errEl.textContent = '';
  const config = readFormValues();

  for (const [k, v] of Object.entries(config)) {
    if (typeof v === 'number' && Number.isNaN(v)) {
      errEl.textContent = `Please provide a valid value for "${k}".`;
      return;
    }
  }

  btn.disabled = true;
  btn.textContent = 'Running...';
  try {
    const result = await apiPost('/simulate', config);
    lastResult = result;

    await playSimulationAnimation(config, result);

    const [execComparison, cpuScaling, gpuScaling, commScaling] = await Promise.all([
      apiPost('/graphs/execution-comparison', config),
      apiPost('/graphs/processor-scaling', { ...config, use_gpu: false }),
      config.gpus > 0
        ? apiPost('/graphs/processor-scaling', { ...config, use_gpu: true })
        : Promise.resolve([]),
      apiPost('/graphs/communication-scaling', config),
    ]);

    goToPage('results');
    renderResults(result);
    updateExecComparisonChart(execComparison);
    updateSpeedupEfficiencyCharts(cpuScaling, gpuScaling);
    updateAmdahlChart(result.amdahl_curve);
    updateCommChart(commScaling);

    const cpuOnlyResult = await apiPost('/simulate', { ...config, gpus: 0, parallelization: 'cpu_multicore' });
    updateCpuGpuChart(cpuOnlyResult, result);

    document.getElementById('amdahl-p-slider').value = config.parallel_fraction;
    document.getElementById('amdahl-p-display').textContent = config.parallel_fraction.toFixed(2);

  } catch (err) {
    errEl.textContent = err.message;
    goToPage('workload');
  } finally {
    btn.disabled = false;
    btn.textContent = '\u25B6 RUN SIMULATION';
  }
}

document.getElementById('run-simulation-btn').addEventListener('click', (e) => {
  e.preventDefault();
  runFullSimulation();
});

// Amdahl slider live-updates the theoretical curve without a full re-run
const amdahlSlider = document.getElementById('amdahl-p-slider');
amdahlSlider.addEventListener('input', async () => {
  const p = parseFloat(amdahlSlider.value);
  document.getElementById('amdahl-p-display').textContent = p.toFixed(2);
  try {
    const curve = await apiPost('/amdahl', { parallel_fraction: p });
    updateAmdahlChart(curve);
  } catch (e) { /* ignore transient errors while dragging */ }
});

// ---------------------------------------------------------------------
// Experiments
// ---------------------------------------------------------------------
const EXPERIMENTS = {
  cores: {
    desc: 'Sweeps CPU cores through 1, 2, 4, 8, 16, 32 with a fixed workload and CPU Multicore strategy. Expect execution time to fall and efficiency to gradually decline.',
    run: async (base) => {
      const points = [1, 2, 4, 8, 16, 32];
      const rows = [];
      for (const n of points) {
        const r = await apiPost('/simulate', { ...base, cpu_cores: n, gpus: 0, parallelization: 'cpu_multicore' });
        rows.push({ x: n, execution_time: r.execution_time, speedup: r.speedup, efficiency: r.efficiency });
      }
      return { rows, xKey: 'x', y1: 'execution_time', y1Label: 'Execution Time (s)', y2: 'efficiency', y2Label: 'Efficiency (%)' };
    },
  },
  gpus: {
    desc: 'Sweeps GPU count through 1, 2, 4, 8 with Data Parallelism. Expect diminishing speedup returns and rising communication overhead.',
    run: async (base) => {
      const points = [1, 2, 4, 8];
      const rows = [];
      for (const n of points) {
        const r = await apiPost('/simulate', { ...base, gpus: n, parallelization: 'data_parallel' });
        rows.push({ x: n, execution_time: r.execution_time, communication_overhead_pct: r.communication_overhead_pct });
      }
      return { rows, xKey: 'x', y1: 'execution_time', y1Label: 'Execution Time (s)', y2: 'communication_overhead_pct', y2Label: 'Communication Overhead (%)' };
    },
  },
  batch: {
    desc: 'Sweeps batch size through 16, 32, 64, 128, 256 with the current hardware configuration. Larger batches better utilize GPUs (up to the reference batch size), changing throughput and efficiency.',
    run: async (base) => {
      const points = [16, 32, 64, 128, 256];
      const rows = [];
      for (const n of points) {
        const r = await apiPost('/simulate', { ...base, batch_size: n });
        rows.push({ x: n, execution_time: r.execution_time, throughput: r.throughput_samples_per_sec });
      }
      return { rows, xKey: 'x', y1: 'execution_time', y1Label: 'Execution Time (s)', y2: 'throughput', y2Label: 'Throughput (samples/s)' };
    },
  },
  fraction: {
    desc: "Sweeps Amdahl's parallelizable fraction P through 0.5, 0.7, 0.8, 0.9, 0.95, 0.99 at a fixed processor count equal to your configured devices. Demonstrates the hard ceiling on speedup imposed by the sequential portion.",
    run: async (base) => {
      const points = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99];
      const rows = [];
      for (const p of points) {
        const r = await apiPost('/simulate', { ...base, parallel_fraction: p });
        rows.push({ x: p, execution_time: r.execution_time, amdahl_speedup: r.amdahl_theoretical_speedup });
      }
      return { rows, xKey: 'x', y1: 'amdahl_speedup', y1Label: "Amdahl's Theoretical Speedup", y2: 'execution_time', y2Label: 'Execution Time (s)' };
    },
  },
};

let currentExperiment = 'cores';

document.querySelectorAll('.exp-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    document.querySelectorAll('.exp-chip').forEach(c => c.classList.remove('active'));
    chip.classList.add('active');
    currentExperiment = chip.dataset.exp;
    document.getElementById('experiment-desc').textContent = EXPERIMENTS[currentExperiment].desc;
  });
});

document.getElementById('run-experiment-btn').addEventListener('click', async () => {
  const btn = document.getElementById('run-experiment-btn');
  btn.disabled = true;
  btn.textContent = 'Running experiment...';
  try {
    const base = readFormValues();
    const exp = EXPERIMENTS[currentExperiment];
    const { rows, xKey, y1, y1Label, y2, y2Label } = await exp.run(base);
    document.getElementById('experiment-results').style.display = 'block';
    updateExperimentCharts(rows, xKey, y1, y1Label, y2, y2Label);
  } catch (err) {
    alert(`Experiment failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Experiment';
  }
});

// Initialize experiment description on load
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('experiment-desc').textContent = EXPERIMENTS.cores.desc;
});
