/**
 * charts.js
 * ----------
 * All Chart.js instances for the Results and Experiments pages. Charts are
 * created once and updated in place on every simulation run, so it's clear
 * that changing inputs actually recomputes what's plotted.
 */

const CHART_COLORS = {
  accent: '#4FD1C5',
  accent2: '#F0B429',
  grid: 'rgba(147,160,194,0.12)',
  text: '#93A0C2',
};

Chart.defaults.font.family = "'IBM Plex Mono', monospace";
Chart.defaults.color = CHART_COLORS.text;
Chart.defaults.borderColor = CHART_COLORS.grid;

const charts = {};

function baseLineOptions(yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400 },
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { color: CHART_COLORS.grid }, ticks: { color: CHART_COLORS.text } },
      y: {
        grid: { color: CHART_COLORS.grid },
        ticks: { color: CHART_COLORS.text },
        title: yLabel ? { display: true, text: yLabel, color: CHART_COLORS.text } : undefined,
        beginAtZero: true,
      },
    },
  };
}

function makeBarChart(ctx, labels, data, label, color) {
  return new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label, data, backgroundColor: color, borderRadius: 3 }] },
    options: baseLineOptions(),
  });
}

function makeLineChart(ctx, labels, data, label, color, yLabel) {
  return new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label, data, borderColor: color, backgroundColor: color,
        tension: 0.25, pointRadius: 3, pointBackgroundColor: color,
      }],
    },
    options: baseLineOptions(yLabel),
  });
}

function initCharts() {
  charts.execComparison = makeBarChart(
    document.getElementById('chart-exec-comparison'), [], [], 'Execution Time (s)', CHART_COLORS.accent
  );
  charts.speedup = makeLineChart(
    document.getElementById('chart-speedup'), [], [], 'Speedup', CHART_COLORS.accent, 'Speedup (x)'
  );
  charts.efficiency = makeLineChart(
    document.getElementById('chart-efficiency'), [], [], 'Efficiency (%)', CHART_COLORS.accent2, 'Efficiency (%)'
  );
  charts.cpuGpu = new Chart(document.getElementById('chart-cpu-gpu'), {
    type: 'bar',
    data: {
      labels: ['Execution Time (s)', 'Throughput (samples/s x0.01)', 'Utilization (%)'],
      datasets: [
        { label: 'CPU only', data: [0, 0, 0], backgroundColor: CHART_COLORS.accent2, borderRadius: 3 },
        { label: 'CPU + GPU', data: [0, 0, 0], backgroundColor: CHART_COLORS.accent, borderRadius: 3 },
      ],
    },
    options: { ...baseLineOptions(), plugins: { legend: { display: true, labels: { color: CHART_COLORS.text } } } },
  });
  charts.amdahl = makeLineChart(
    document.getElementById('chart-amdahl'), [], [], "Amdahl's Law Speedup", CHART_COLORS.accent, 'Speedup (x)'
  );
  charts.comm = new Chart(document.getElementById('chart-comm'), {
    type: 'bar',
    data: {
      labels: [],
      datasets: [{ label: 'Communication Overhead (%)', data: [], backgroundColor: CHART_COLORS.accent2, borderRadius: 3 }],
    },
    options: baseLineOptions('Overhead (%)'),
  });

  charts.exp1 = makeLineChart(document.getElementById('chart-exp-1'), [], [], 'Execution Time (s)', CHART_COLORS.accent, 'Seconds');
  charts.exp2 = makeLineChart(document.getElementById('chart-exp-2'), [], [], 'Speedup', CHART_COLORS.accent2, 'Value');
}

function updateExecComparisonChart(rows) {
  charts.execComparison.data.labels = rows.map(r => r.label);
  charts.execComparison.data.datasets[0].data = rows.map(r => r.execution_time);
  charts.execComparison.update();
}

function updateSpeedupEfficiencyCharts(cpuRows, gpuRows) {
  // Use GPU-scaling series when GPUs are configured, else CPU-scaling.
  const rows = gpuRows && gpuRows.length ? gpuRows : cpuRows;
  charts.speedup.data.labels = rows.map(r => r.processors);
  charts.speedup.data.datasets[0].data = rows.map(r => r.speedup);
  charts.speedup.update();

  charts.efficiency.data.labels = rows.map(r => r.processors);
  charts.efficiency.data.datasets[0].data = rows.map(r => r.efficiency);
  charts.efficiency.update();
}

function updateCpuGpuChart(cpuResult, gpuResult) {
  charts.cpuGpu.data.datasets[0].data = [
    cpuResult.execution_time,
    cpuResult.throughput_samples_per_sec * 0.01,
    cpuResult.cpu_utilization,
  ];
  charts.cpuGpu.data.datasets[1].data = [
    gpuResult.execution_time,
    gpuResult.throughput_samples_per_sec * 0.01,
    Math.max(gpuResult.cpu_utilization, gpuResult.gpu_utilization),
  ];
  charts.cpuGpu.update();
}

function updateAmdahlChart(curve) {
  charts.amdahl.data.labels = curve.map(p => p.processors);
  charts.amdahl.data.datasets[0].data = curve.map(p => p.speedup);
  charts.amdahl.update();
}

function updateCommChart(rows) {
  charts.comm.data.labels = rows.map(r => r.devices);
  charts.comm.data.datasets[0].data = rows.map(r => r.communication_overhead_pct);
  charts.comm.update();
}

function updateExperimentCharts(rows, xKey, y1Key, y1Label, y2Key, y2Label) {
  charts.exp1.data.labels = rows.map(r => r[xKey]);
  charts.exp1.data.datasets[0].data = rows.map(r => r[y1Key]);
  charts.exp1.data.datasets[0].label = y1Label;
  charts.exp1.update();
  document.getElementById('exp-chart1-title').textContent = y1Label;

  charts.exp2.data.labels = rows.map(r => r[xKey]);
  charts.exp2.data.datasets[0].data = rows.map(r => r[y2Key]);
  charts.exp2.data.datasets[0].label = y2Label;
  charts.exp2.update();
  document.getElementById('exp-chart2-title').textContent = y2Label;
}

document.addEventListener('DOMContentLoaded', initCharts);
