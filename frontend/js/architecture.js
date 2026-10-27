/**
 * architecture.js
 * ----------------
 * Builds the educational diagrams on the "Architecture" page: the overall
 * pipeline, an SIMD scalar-vs-vector demo, the memory hierarchy, and a
 * data/model parallelism illustration. Pure DOM rendering, no dependencies.
 */

function renderArchitectureDiagram() {
  const el = document.getElementById('architecture-diagram');
  el.innerHTML = `
    <div class="arch-node">AI Workload (Deep Learning training job)</div>
    <div class="arch-arrow">&#8595;</div>
    <div class="arch-node">Workload Analyzer<br><span style="color:var(--text-faint); font-size:11px;">dataset, model size, batches &rarr; total operations</span></div>
    <div class="arch-arrow">&#8595;</div>
    <div class="arch-node highlight">Parallelization Engine</div>
    <div class="arch-arrow">&#8595;</div>
    <div class="arch-branch">
      <div class="arch-node">CPU Processing</div>
      <div class="arch-node">GPU Processing</div>
      <div class="arch-node">Distributed Processing</div>
    </div>
    <div class="arch-arrow">&#8595;</div>
    <div class="arch-node">Memory Model</div>
    <div class="arch-arrow">&#8595;</div>
    <div class="arch-node">Communication Overhead</div>
    <div class="arch-arrow">&#8595;</div>
    <div class="arch-node highlight">Performance Model</div>
    <div class="arch-arrow">&#8595;</div>
    <div class="arch-branch">
      <div class="arch-node">Time</div>
      <div class="arch-node">Speedup</div>
      <div class="arch-node">Efficiency</div>
    </div>
    <div class="arch-arrow">&#8595;</div>
    <div class="arch-node">Visual Results Dashboard</div>
  `;
}

function renderSimdDemo() {
  const el = document.getElementById('simd-demo');
  const vals = [2, 3, 4, 5];
  const cells = (arr) => arr.map(v => `<div class="simd-cell">${v}</div>`).join('');
  el.innerHTML = `
    <div style="margin-bottom:14px;">
      <div style="color:var(--text-faint); margin-bottom:6px;">Scalar (one operation at a time)</div>
      ${vals.map(v => `<div class="simd-row"><span>${v} + ${v} = ${v + v}</span></div>`).join('')}
    </div>
    <div>
      <div style="color:var(--text-faint); margin-bottom:6px;">Vector (SIMD — one instruction, many elements)</div>
      <div class="simd-row">${cells(vals)} <span style="color:var(--text-faint);">+</span> ${cells(vals)} <span style="color:var(--text-faint);">=</span> ${cells(vals.map(v => v + v))}</div>
    </div>
  `;
}

function renderMemoryHierarchy() {
  const el = document.getElementById('memory-hierarchy');
  const tiers = [
    { name: 'Registers', tag: 'fastest · smallest' },
    { name: 'Cache (L1/L2/L3)', tag: 'fast · small' },
    { name: 'RAM', tag: 'moderate speed · large' },
    { name: 'GPU Memory', tag: 'high bandwidth · limited capacity' },
    { name: 'Storage (Disk/SSD)', tag: 'slowest · largest' },
  ];
  el.innerHTML = tiers.map(t => `
    <div class="mem-tier"><span class="mem-name">${t.name}</span><span class="mem-tag">${t.tag}</span></div>
  `).join('');
}

function renderParallelDemos() {
  const dataEl = document.getElementById('data-parallel-demo');
  dataEl.innerHTML = `
    <div style="margin-bottom:8px; color:var(--text-faint);">Dataset</div>
    <div class="device-row"><div class="device-bar-track"><div class="device-bar-fill" style="width:100%"></div></div></div>
    <div style="margin:10px 0 8px 0; color:var(--text-faint);">split across devices &darr;</div>
    ${[1,2,3,4].map(i => `
      <div class="device-row">
        <span class="device-label">GPU ${i}</span>
        <div class="device-bar-track"><div class="device-bar-fill" style="width:25%"></div></div>
      </div>`).join('')}
  `;

  const modelEl = document.getElementById('model-parallel-demo');
  modelEl.innerHTML = `
    <div class="device-row"><span class="device-label">GPU 1</span>
      <div class="device-bar-track"><div class="device-bar-fill" style="width:100%"></div></div>
      <span style="color:var(--text-faint); margin-left:8px;">Layers 1-3</span>
    </div>
    <div class="device-row"><span class="device-label">GPU 2</span>
      <div class="device-bar-track"><div class="device-bar-fill" style="width:100%; background:var(--accent-2);"></div></div>
      <span style="color:var(--text-faint); margin-left:8px;">Layers 4-6</span>
    </div>
    <div style="color:var(--text-faint); margin-top:10px;">Activations flow GPU 1 &rarr; GPU 2 between layer groups.</div>
  `;
}

function initArchitecturePage() {
  renderArchitectureDiagram();
  renderSimdDemo();
  renderMemoryHierarchy();
  renderParallelDemos();
}

document.addEventListener('DOMContentLoaded', initArchitecturePage);
