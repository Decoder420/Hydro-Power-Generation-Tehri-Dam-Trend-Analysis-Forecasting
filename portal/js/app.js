/**
 * National Hydroelectric Operations & Grid Dispatch Portal (NHOP-DSS)
 * Interactive logic, telemetry charts, storm simulator & DSM calculation.
 */

document.addEventListener('DOMContentLoaded', async () => {
  // Update live clock
  function updateClock() {
    const now = new Date();
    const opts = { timeZone: 'Asia/Kolkata', hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' };
    const dateOpts = { timeZone: 'Asia/Kolkata', year: 'numeric', month: 'short', day: '2-digit' };
    const timeStr = now.toLocaleTimeString('en-IN', opts);
    const dateStr = now.toLocaleDateString('en-IN', dateOpts);
    const el = document.getElementById('live-ist-clock');
    if (el) el.textContent = `${dateStr} | ${timeStr} IST`;
  }
  setInterval(updateClock, 1000);
  updateClock();

  // Tab Navigation
  const tabBtns = document.querySelectorAll('.nav-tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add('active');
    });
  });

  // Load telemetry data
  let telemetry = null;
  try {
    const res = await fetch('data/telemetry.json');
    telemetry = await res.json();
    initPortal(telemetry);
  } catch (err) {
    console.warn('Failed to fetch telemetry.json via HTTP, using fallback simulation:', err);
    initFallback();
  }
});

function initPortal(data) {
  const cur = data.current;

  // 1. Populate KPI Cards
  document.getElementById('kpi-res-elev').textContent = `${cur.reservoir_level.toFixed(2)} m`;
  document.getElementById('kpi-res-storage').textContent = `${cur.live_storage_pct.toFixed(1)}%`;
  document.getElementById('kpi-res-storage-mcm').textContent = `${cur.storage_mcm.toFixed(0)} / 3540 MCM`;
  document.getElementById('kpi-power').textContent = `${cur.power_mw.toFixed(1)} MW`;
  document.getElementById('kpi-inflow').textContent = `${cur.inflow_m3_s.toFixed(1)} m³/s`;
  document.getElementById('kpi-water-rate').textContent = `${Math.round(cur.water_rate)} m³/MWh`;
  document.getElementById('kpi-head').textContent = `Net Head: ${cur.net_head_m.toFixed(1)} m`;

  // 2. Animate Reservoir Gauge
  // Elevation ranges from 740 (0%) to 830 (100%)
  const minElev = 740.0;
  const maxElev = 830.0;
  const fillPct = Math.max(5, Math.min(95, ((cur.reservoir_level - minElev) / (maxElev - minElev)) * 100));
  const waterBody = document.getElementById('water-body');
  if (waterBody) {
    waterBody.style.height = `${fillPct}%`;
  }
  const currentMarker = document.getElementById('marker-current');
  if (currentMarker) {
    currentMarker.style.bottom = `${fillPct}%`;
    currentMarker.textContent = `Current: ${cur.reservoir_level.toFixed(1)}m`;
  }

  // 3. Render 4 Turbine Units SCADA Board
  const unitsContainer = document.getElementById('units-grid');
  if (unitsContainer && cur.units) {
    unitsContainer.innerHTML = cur.units.map(u => `
      <div class="unit-card">
        <div class="unit-header">${u.name}</div>
        <span class="unit-status ${u.status === 'RUNNING' ? 'status-running' : 'status-standby'}">${u.status}</span>
        <div class="unit-mw">${u.mw} <span style="font-size:0.8rem; font-weight:600;">MW</span></div>
        <div class="unit-detail">Turbine Q: <strong>${u.flow_m3_s} m³/s</strong></div>
        <div class="unit-detail">Bearing Temp: <strong>${u.temp_c} °C</strong></div>
        <div class="unit-detail">Vibration: <strong>${u.vibration_mm_s} mm/s</strong></div>
      </div>
    `).join('');
  }

  // 4. Render Catchment Stations Table
  const stationsBody = document.getElementById('stations-table-body');
  if (stationsBody && data.stations) {
    stationsBody.innerHTML = data.stations.map(s => `
      <tr>
        <td><strong>${s.name}</strong></td>
        <td>${s.elevation} m</td>
        <td>${s.distance_km} km</td>
        <td>${s.level_m.toFixed(1)} m</td>
        <td>${s.discharge_m3_s.toFixed(1)} m³/s</td>
        <td><span class="status-badge" style="background:#dcfce7; color:#15803d; font-weight:700;">${s.status}</span></td>
      </tr>
    `).join('');
  }

  // 5. Initialize Charts
  initScheduleChart(data.schedule_96);
  initHistoricalCharts(data.history);
  initStormSimulator(cur.reservoir_level);
  initDsmCalculator();
}

function initScheduleChart(scheduleData) {
  const ctx = document.getElementById('chart-96-schedule');
  if (!ctx) return;

  const labels = scheduleData.map(d => d.Time);
  const scheduled = scheduleData.map(d => d.Scheduled_MW);
  const dsmLow = scheduleData.map(d => d.DSM_Band_Low_MW);
  const dsmHigh = scheduleData.map(d => d.DSM_Band_High_MW);

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'NRLDC Scheduled MW',
          data: scheduled,
          borderColor: '#0b2545',
          backgroundColor: 'rgba(11, 37, 69, 0.1)',
          borderWidth: 2.5,
          fill: true,
          tension: 0.2
        },
        {
          label: 'DSM Upper Band (+12%)',
          data: dsmHigh,
          borderColor: '#f59e0b',
          borderDash: [5, 5],
          borderWidth: 1.5,
          fill: false,
          pointRadius: 0
        },
        {
          label: 'DSM Lower Band (-12%)',
          data: dsmLow,
          borderColor: '#ef4444',
          borderDash: [5, 5],
          borderWidth: 1.5,
          fill: false,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top' },
        tooltip: { mode: 'index', intersect: false }
      },
      scales: {
        x: { ticks: { maxTicksLimit: 16 } },
        y: { title: { display: true, text: 'Power Output (MW)' }, min: 0, max: 1050 }
      }
    }
  });
}

function initHistoricalCharts(history) {
  const ctxTrend = document.getElementById('chart-history-trend');
  if (ctxTrend) {
    new Chart(ctxTrend, {
      type: 'line',
      data: {
        labels: history.dates,
        datasets: [
          {
            label: 'River Inflow (m³/s)',
            data: history.inflow,
            borderColor: '#0077b6',
            borderWidth: 1.5,
            pointRadius: 0,
            yAxisID: 'y'
          },
          {
            label: 'Power Output (MW)',
            data: history.power,
            borderColor: '#10b981',
            borderWidth: 1.5,
            pointRadius: 0,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'top' } },
        scales: {
          x: { ticks: { maxTicksLimit: 12 } },
          y: { type: 'linear', position: 'left', title: { display: true, text: 'Inflow (m³/s)' } },
          y1: { type: 'linear', position: 'right', grid: { drawOnChartArea: false }, title: { display: true, text: 'Power Output (MW)' } }
        }
      }
    });
  }

  const ctxElevation = document.getElementById('chart-reservoir-elevation');
  if (ctxElevation) {
    new Chart(ctxElevation, {
      type: 'line',
      data: {
        labels: history.dates,
        datasets: [
          {
            label: 'Reservoir Level (m MSL)',
            data: history.elevation,
            borderColor: '#0284c7',
            backgroundColor: 'rgba(2, 132, 199, 0.15)',
            borderWidth: 2,
            fill: true,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          annotation: {
            annotations: {
              frl: { type: 'line', yMin: 830, yMax: 830, borderColor: '#ef4444', borderWidth: 2, label: { content: 'FRL (830m)', enabled: true } }
            }
          }
        },
        scales: {
          x: { ticks: { maxTicksLimit: 12 } },
          y: { min: 730, max: 840, title: { display: true, text: 'Elevation (m MSL)' } }
        }
      }
    });
  }
}

function initStormSimulator(initialElev) {
  const rainSlider = document.getElementById('sim-rain-input');
  const rainVal = document.getElementById('sim-rain-val');
  const turbSlider = document.getElementById('sim-turb-input');
  const turbVal = document.getElementById('sim-turb-val');
  const simTableBody = document.getElementById('sim-table-body');

  function updateSimulation() {
    const rain = parseFloat(rainSlider.value);
    const turbQ = parseFloat(turbSlider.value);
    rainVal.textContent = `${rain} mm`;
    turbVal.textContent = `${turbQ} m³/s`;

    const FRL = 830.0;
    const deadStorage = 925.0;
    const liveStorage = 2615.0;

    // Hypsometric helper
    function elevToStorage(z) {
      const frac = Math.max(0, Math.min(1.1, (z - 740.0) / 90.0));
      return deadStorage + liveStorage * Math.pow(frac, 1.35);
    }
    function storageToElev(s) {
      const live = Math.max(0, s - deadStorage);
      const frac = Math.pow(live / liveStorage, 1.0 / 1.35);
      return 740.0 + frac * 90.0;
    }

    let curStorage = elevToStorage(initialElev);
    const M3_S_TO_MCM = 0.0864;
    const stormDist = [rain * 0.35, rain * 0.40, rain * 0.15, rain * 0.10, 0, 0, 0];

    let rowsHtml = '';
    let totalSpill = 0;
    let totalEnergyGWh = 0;

    for (let day = 1; day <= 7; day++) {
      const r = stormDist[day - 1];
      const stormInflow = (r * 32.0) + 110.0;
      const inflowMcm = stormInflow * M3_S_TO_MCM;
      const turbMcm = Math.min(turbQ, 500.0) * M3_S_TO_MCM;

      let netStorage = curStorage + inflowMcm - turbMcm - 0.45;
      const maxStorage = elevToStorage(FRL);

      let spillQ = 0;
      if (netStorage > maxStorage) {
        const excessMcm = netStorage - maxStorage;
        spillQ = excessMcm / M3_S_TO_MCM;
        netStorage = maxStorage;
      }
      curStorage = netStorage;
      const elev = storageToElev(curStorage);

      // Power calculation: MW ~ 0.88 * 9.81 * Q * H / 1000
      const head = Math.max(10, elev - 598.0);
      const powerMw = Math.min(1000.0, (0.88 * 9.80665 * turbQ * head) / 1000.0);
      const gwh = (powerMw * 24.0) / 1000.0;

      totalSpill += spillQ;
      totalEnergyGWh += gwh;

      rowsHtml += `
        <tr>
          <td><strong>Day +${day}</strong></td>
          <td>${r.toFixed(1)} mm</td>
          <td>${stormInflow.toFixed(1)} m³/s</td>
          <td>${turbQ.toFixed(1)} m³/s</td>
          <td style="color:${spillQ > 0 ? '#ef4444' : '#15803d'}; font-weight:700;">${spillQ.toFixed(1)} m³/s</td>
          <td><strong>${elev.toFixed(2)} m</strong></td>
          <td>${powerMw.toFixed(1)} MW</td>
          <td>${gwh.toFixed(2)} GWh</td>
        </tr>
      `;
    }

    if (simTableBody) simTableBody.innerHTML = rowsHtml;
    const simSpillBadge = document.getElementById('sim-total-spill');
    if (simSpillBadge) {
      simSpillBadge.textContent = totalSpill > 0 ? `Spillway Engaged: ${totalSpill.toFixed(1)} m³/s` : 'Safe (Zero Spilling)';
      simSpillBadge.style.color = totalSpill > 0 ? '#ef4444' : '#15803d';
    }
  }

  if (rainSlider && turbSlider) {
    rainSlider.addEventListener('input', updateSimulation);
    turbSlider.addEventListener('input', updateSimulation);
    updateSimulation();
  }
}

function initDsmCalculator() {
  const schedMwInput = document.getElementById('dsm-sched-mw');
  const actMwInput = document.getElementById('dsm-act-mw');
  const freqInput = document.getElementById('dsm-grid-freq');
  const dsmResultEl = document.getElementById('dsm-result-box');

  function calcDsm() {
    const sched = parseFloat(schedMwInput.value) || 0;
    const actual = parseFloat(actMwInput.value) || 0;
    const freq = parseFloat(freqInput.value) || 50.0;

    const dev = actual - sched;
    const devPct = sched > 0 ? (dev / sched) * 100.0 : 0;

    // IEGC 2023 DSM rate: Base ~ ₹2.50/kWh (₹2500/MWh) scaled by frequency
    // When frequency is low (<50.0 Hz), under-injection is heavily penalized
    let dsmRate = 3500; // Rs/MWh
    if (freq < 49.90) dsmRate = 8000;
    else if (freq > 50.05) dsmRate = 1200;

    // 15-minute energy deviation (MWh) = MW * 0.25
    const devEnergyMwh = dev * 0.25;
    const penaltyOrIncentive = devEnergyMwh * dsmRate;

    let status = 'COMPLIANT (WITHIN ±12% TOLERANCE)';
    let color = '#15803d';

    if (Math.abs(devPct) > 12.0) {
      if (dev < 0) {
        status = 'VIOLATION: UNDER-INJECTION PENALTY';
        color = '#ef4444';
      } else {
        status = 'OVER-INJECTION (ENERGY SURPLUS)';
        color = '#b45309';
      }
    }

    if (dsmResultEl) {
      dsmResultEl.innerHTML = `
        <div style="font-size:0.85rem; color:${color}; font-weight:700; margin-bottom:6px;">${status}</div>
        <div style="font-size:1.1rem; font-weight:800; color:var(--gov-navy);">
          Deviation: ${dev > 0 ? '+' : ''}${dev.toFixed(1)} MW (${devPct.toFixed(1)}%)
        </div>
        <div style="font-size:0.82rem; color:#475569; margin-top:4px;">
          Estimated DSM Cash Settlement: <strong>₹${Math.abs(penaltyOrIncentive).toLocaleString('en-IN', { maximumFractionDigits: 0 })}</strong>
          (${dev >= 0 ? 'Receivable' : 'Payable to Pool'})
        </div>
      `;
    }
  }

  if (schedMwInput && actMwInput && freqInput) {
    schedMwInput.addEventListener('input', calcDsm);
    actMwInput.addEventListener('input', calcDsm);
    freqInput.addEventListener('input', calcDsm);
    calcDsm();
  }
}
