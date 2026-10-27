/**
 * help-content.js
 * ----------------
 * Plain-language explanations shown in the "What does this mean?" modal
 * for each metric / graph in the results dashboard.
 */
const HELP_CONTENT = {
  exec_time: {
    title: "Execution Time",
    body: "The estimated time to complete training under the chosen configuration. It is calculated from the total computational work divided by effective CPU/GPU throughput, plus any communication and memory penalties. This is a modeled estimate, not a measurement from real hardware."
  },
  speedup: {
    title: "Speedup",
    body: "Speedup tells us how many times faster the parallel configuration is compared with a single-core sequential baseline. Speedup = Baseline Execution Time / Parallel Execution Time."
  },
  efficiency: {
    title: "Parallel Efficiency",
    body: "Efficiency tells us how effectively the available processing units are being used. Efficiency = (Speedup / Number of Processing Units) x 100. Perfect linear scaling would give 100%; overhead pulls it lower."
  },
  throughput: {
    title: "Throughput",
    body: "Throughput measures how many training samples the system processes per second: Throughput = Samples Processed / Execution Time. Higher throughput means more data is processed in the same amount of time."
  },
  comm: {
    title: "Communication Overhead",
    body: "Communication overhead is the time spent transferring information (like gradients) between processors or devices instead of performing useful computation. It grows with more devices, larger models, lower bandwidth, and higher latency."
  },
  exec_compare: {
    title: "Execution Time Comparison",
    body: "Compares estimated execution time across five configurations of the same workload: sequential, CPU multicore, single GPU, multi-GPU, and a distributed cluster. It shows the practical impact of each HPC technique on the same job."
  },
  speedup_curve: {
    title: "Speedup vs. Processors",
    body: "Shows how speedup changes as the number of CPU cores or GPUs increases, using your configured workload. Growth typically slows down as overhead accumulates — this is diminishing returns."
  },
  efficiency_curve: {
    title: "Efficiency vs. Processors",
    body: "Shows how parallel efficiency changes as more processing units are added. Efficiency usually declines as processor count grows, because synchronization and communication overhead consume a larger share of the total time."
  },
  amdahl: {
    title: "Amdahl's Law",
    body: "Amdahl's Law shows that the maximum speedup of a program is limited by the portion of it that cannot be parallelized. Speedup = 1 / ((1 - P) + (P / N)), where P is the parallelizable fraction and N is the number of processors. Even with unlimited processors, speedup is capped at 1 / (1 - P)."
  },
  comm_curve: {
    title: "Communication Overhead vs. Devices",
    body: "Shows how communication time (in seconds and as a percentage of total execution time) grows as more GPUs or nodes are added to the parallel configuration, for your selected parallelization strategy."
  },
  data_parallel: {
    title: "Data Parallelism",
    body: "Data parallelism divides the dataset across multiple processors or GPUs. Each device holds a full copy of the model and processes a different slice of data; results (gradients) are synchronized periodically."
  },
  model_parallel: {
    title: "Model Parallelism",
    body: "Model parallelism divides different parts (e.g. layers) of a single model across multiple devices. It's used when a model is too large to fit on one device's memory."
  }
};

function openHelpModal(key) {
  const content = HELP_CONTENT[key];
  if (!content) return;
  document.getElementById('help-modal-title').textContent = content.title;
  document.getElementById('help-modal-body').textContent = content.body;
  document.getElementById('help-modal').classList.add('open');
}

document.addEventListener('DOMContentLoaded', () => {
  document.body.addEventListener('click', (e) => {
    const btn = e.target.closest('.help-btn');
    if (btn) {
      e.preventDefault();
      openHelpModal(btn.dataset.help);
    }
  });
  document.getElementById('help-modal-close').addEventListener('click', () => {
    document.getElementById('help-modal').classList.remove('open');
  });
  document.getElementById('help-modal').addEventListener('click', (e) => {
    if (e.target.id === 'help-modal') e.target.classList.remove('open');
  });
});
