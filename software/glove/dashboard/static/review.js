// Review UI — <img> scrubbing against pre-extracted JPEG cache.
// Server runs ffmpeg in the background to populate <session>/cache/scrub_640/;
// while that's warming, PyAV serves individual frames (slow but functional).
//
// Keys: a=approve · r=reject · f=force_include · u=clear · s=skip · [/]=prev/next demo
//       ←/→=step 1 frame · shift+←/→=step 10 · n/p=flagged frame · space=play/pause

(function () {
  const data = window.REVIEW_PANEL;
  const sessionId = window.SESSION_ID;
  const LIGHT = {
    text: '#1c1917', dim: '#57534e', muted: '#a8a29e', border: '#e7e5e4',
    accent: '#0c4a6e', pass: '#15803d', warn: '#b45309', fail: '#b91c1c',
  };
  const PLOTLY_LAYOUT_BASE = {
    paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
    font: { family: 'JetBrains Mono, SF Mono, monospace', size: 10, color: LIGHT.dim },
    margin: { l: 42, r: 12, t: 6, b: 28 },
    showlegend: false,
    xaxis: { gridcolor: LIGHT.border, zerolinecolor: LIGHT.border, linecolor: LIGHT.border },
    yaxis: { gridcolor: LIGHT.border, zerolinecolor: LIGHT.border, linecolor: LIGHT.border },
  };
  const PLOT_CONFIG = { displaylogo: false, responsive: true, displayModeBar: false };

  const nFrames = (data && data.n_frames) || 0;
  const fps = (data && data.fps) || 60.0;
  let currentFrame = 0;

  const img = document.getElementById('tr-frame-img');
  const slider = document.getElementById('tr-frame-slider');
  const counter = document.getElementById('tr-frame-counter');
  const timeLabel = document.getElementById('tr-frame-time');
  const scrubStatus = document.getElementById('tr-scrub-status');

  // ---------- Frame loading with stale-request drop ----------
  let loadSeq = 0;
  function loadFrameIntoDisplay(idx) {
    if (!img || !sessionId) return;
    const mySeq = ++loadSeq;
    const pre = new Image();
    pre.onload = () => {
      if (mySeq !== loadSeq) return;   // newer request already won
      img.src = pre.src;
    };
    pre.onerror = () => {};
    pre.src = `/review/${sessionId}/frame/${idx}`;
  }

  function setFrame(idx, { updateSlider = true } = {}) {
    if (nFrames === 0) return;
    idx = Math.max(0, Math.min(nFrames - 1, Math.round(idx)));
    currentFrame = idx;
    if (updateSlider && slider) slider.value = String(idx);
    if (counter) counter.textContent = `${idx} / ${Math.max(0, nFrames - 1)}`;
    if (timeLabel) timeLabel.textContent = ` · ${(idx / fps).toFixed(2)}s`;
    updateCursors(idx);
    loadFrameIntoDisplay(idx);
  }

  // ---------- Plot cursors: vertical line on time plots, dot on trajectory ----------
  function trajIndexForFrame(frame) {
    const xy = data && data.pose_xy;
    if (!xy || !xy[0] || !xy[0].length) return 0;
    const ds_len = xy[0].length;
    if (nFrames <= 1) return 0;
    return Math.max(0, Math.min(ds_len - 1, Math.round(frame * (ds_len - 1) / (nFrames - 1))));
  }
  function updateCursors(frame) {
    const tSec = frame / fps;
    for (const id of ['tr-jumps', 'tr-proprio']) {
      const el = document.getElementById(id);
      if (!el || !el.layout) continue;
      try {
        Plotly.relayout(el, {
          shapes: [{
            type: 'line', xref: 'x', yref: 'paper',
            x0: tSec, x1: tSec, y0: 0, y1: 1,
            line: { color: LIGHT.fail, width: 1 }, opacity: 0.75,
          }],
        });
      } catch (_) {}
    }
    const trajEl = document.getElementById('tr-traj');
    if (trajEl && trajEl.data && trajEl.data.length >= 3 && data && data.pose_xy) {
      const j = trajIndexForFrame(frame);
      // Identity display: align.py now expresses pose_* in the ArUco tag frame
      // (gravity-aligned by construction — tag lies flat on the table). No axis
      // remap is required; Z is true height, X/Y lie in the table plane.
      const cx = data.pose_xy[0][j];
      const cy = data.pose_xy[1][j];
      const cz = data.pose_xy[2] ? data.pose_xy[2][j] : 0;
      try { Plotly.restyle(trajEl, { x: [[cx]], y: [[cy]], z: [[cz]] }, [2]); } catch (_) {}
    }
  }

  // ---------- Plotly renders ----------
  function renderTrajectory(d) {
    const el = document.getElementById('tr-traj');
    if (!el) return;
    if (!d) { el.innerHTML = '<div class="tr-empty">no episode data</div>'; return; }
    const [x, y, z] = d.pose_xy;
    const lost = d.is_lost_mask;
    const xg = x.map((v, i) => (lost[i] ? null : v));
    const yg = y.map((v, i) => (lost[i] ? null : v));
    const zg = z.map((v, i) => (lost[i] ? null : v));
    // Time gradient gives the viewer an immediate sense of motion direction.
    const n = x.length;
    const tColor = Array.from({ length: n }, (_, i) => i / Math.max(1, n - 1));

    const trace = {
      type: 'scatter3d',
      x: xg, y: yg, z: zg,
      mode: 'lines+markers',
      line: { color: LIGHT.accent, width: 4 },
      marker: { size: 2, color: tColor, colorscale: 'Viridis', opacity: 0.7, showscale: false },
      hovertemplate: 'x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f} m<extra></extra>',
    };
    const origin = {
      type: 'scatter3d', x: [0], y: [0], z: [0], mode: 'markers',
      marker: { size: 5, color: '#525252', symbol: 'cross' },
      hovertemplate: 'origin<extra></extra>',
    };
    const cursor = {
      type: 'scatter3d',
      x: [x[0] || 0], y: [y[0] || 0], z: [z[0] || 0],
      mode: 'markers',
      marker: { size: 8, color: LIGHT.fail, symbol: 'circle', line: { color: '#fff', width: 2 } },
      hovertemplate: 'frame cursor<extra></extra>',
    };

    const axisCfg = {
      backgroundcolor: '#fafaf9', gridcolor: LIGHT.border, zerolinecolor: LIGHT.border_strong || '#d6d3d1',
      showbackground: true, linecolor: LIGHT.border,
      title: { font: { size: 10, color: LIGHT.dim } },
      tickfont: { size: 9, color: LIGHT.dim },
    };
    const layout = {
      ...PLOTLY_LAYOUT_BASE,
      margin: { l: 0, r: 0, t: 0, b: 0 },
      scene: {
        xaxis: { ...axisCfg, title: { text: 'x (m)', font: { size: 10 } } },
        yaxis: { ...axisCfg, title: { text: 'y (m)', font: { size: 10 } } },
        zaxis: { ...axisCfg, title: { text: 'z (m)', font: { size: 10 } } },
        aspectmode: 'data',
        camera: { eye: { x: 1.35, y: -1.35, z: 0.9 }, up: { x: 0, y: 0, z: 1 } },
      },
    };
    Plotly.newPlot(el, [trace, origin, cursor], layout, PLOT_CONFIG);
  }
  function renderJumps(d) {
    const el = document.getElementById('tr-jumps');
    if (!el) return;
    if (!d) { el.innerHTML = '<div class="tr-empty">no episode data</div>'; return; }
    const jumps = d.pose_jump_mm;
    const t = d.t_rel;
    const thresh = d.pose_jump_threshold_mm || 20;
    const colors = jumps.map(v => (v > thresh ? LIGHT.fail : LIGHT.accent));
    const bar = {
      x: t, y: jumps, type: 'bar', marker: { color: colors },
      hovertemplate: 't=%{x:.2f}s · frame %{customdata}<br>Δpose=%{y:.1f} mm<extra></extra>',
      customdata: t.map(tt => Math.round(tt * fps)),
    };
    const threshLine = { x: [t[0], t[t.length - 1]], y: [thresh, thresh], mode: 'lines',
      line: { color: LIGHT.warn, width: 1, dash: 'dot' }, hoverinfo: 'skip' };
    const layout = {
      ...PLOTLY_LAYOUT_BASE, margin: { l: 42, r: 12, t: 6, b: 22 },
      xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: { text: 't (s)', font: { size: 9 } } },
      yaxis: { ...PLOTLY_LAYOUT_BASE.yaxis, title: { text: 'Δpose (mm)', font: { size: 9 } }, rangemode: 'tozero' },
      bargap: 0.1,
    };
    Plotly.newPlot(el, [bar, threshLine], layout, PLOT_CONFIG);
  }
  function renderProprio(d) {
    const el = document.getElementById('tr-proprio');
    if (!el) return;
    if (!d) { el.innerHTML = '<div class="tr-empty">no episode data</div>'; return; }
    const t = d.t_rel;
    const hallColors = ['#0c4a6e', '#0369a1', '#0284c7', '#0ea5e9'];
    const traces = [];
    for (let i = 0; i < 4; i++) {
      if (d.hall_angles[i] && d.hall_angles[i].length) {
        traces.push({ x: t, y: d.hall_angles[i], xaxis: 'x1', yaxis: 'y1',
          mode: 'lines', line: { color: hallColors[i], width: 1 }, name: `hall ${i}`,
          hovertemplate: `h${i}: %{y:.2f} rad<extra></extra>` });
      }
      if (d.hall_bz[i] && d.hall_bz[i].length) {
        traces.push({ x: t, y: d.hall_bz[i], xaxis: 'x1', yaxis: 'y2',
          mode: 'lines', line: { color: hallColors[i], width: 1, dash: 'dot' }, name: `Bz ${i}`,
          hovertemplate: `Bz${i}: %{y:.0f}<extra></extra>`, opacity: 0.55 });
      }
    }
    if (d.tof_top && d.tof_top.length) {
      traces.push({ x: t, y: d.tof_top, xaxis: 'x1', yaxis: 'y3',
        mode: 'lines', line: { color: LIGHT.warn, width: 1.2 }, name: 'ToF top',
        hovertemplate: 'ToF top: %{y:.0f} mm<extra></extra>' });
    }
    if (d.tof_bottom && d.tof_bottom.length) {
      traces.push({ x: t, y: d.tof_bottom, xaxis: 'x1', yaxis: 'y3',
        mode: 'lines', line: { color: LIGHT.fail, width: 1.2, dash: 'dash' }, name: 'ToF bot',
        hovertemplate: 'ToF bot: %{y:.0f} mm<extra></extra>' });
    }
    const layout = {
      ...PLOTLY_LAYOUT_BASE, margin: { l: 46, r: 42, t: 8, b: 26 },
      showlegend: true, legend: { orientation: 'h', y: -0.22, font: { size: 9 }, bgcolor: 'transparent' },
      grid: { rows: 3, columns: 1, subplots: [['xy'], ['xy2'], ['xy3']], pattern: 'independent' },
      xaxis: { ...PLOTLY_LAYOUT_BASE.xaxis, title: { text: 't (s)', font: { size: 9 } } },
      yaxis:  { ...PLOTLY_LAYOUT_BASE.yaxis, domain: [0.68, 1.0], title: { text: 'hall θ', font: { size: 9 } } },
      yaxis2: { ...PLOTLY_LAYOUT_BASE.yaxis, domain: [0.36, 0.64], title: { text: 'Bz', font: { size: 9 } } },
      yaxis3: { ...PLOTLY_LAYOUT_BASE.yaxis, domain: [0.00, 0.32], title: { text: 'ToF (mm)', font: { size: 9 } } },
    };
    Plotly.newPlot(el, traces, layout, PLOT_CONFIG);
  }
  function wireClickToFrame(elId) {
    const el = document.getElementById(elId);
    if (!el || !el.on) return;
    el.on('plotly_click', (e) => {
      if (!e || !e.points || !e.points.length) return;
      const t = e.points[0].x;
      if (typeof t === 'number') setFrame(Math.round(t * fps));
    });
  }

  // ---------- Flagged-frame iteration ----------
  let flagCursor = -1;
  function flaggedFrames() { return (data && data.flagged_frames) || []; }
  function flaggedMag() { return (data && data.flagged_mag) || []; }
  function ensureIndicator() {
    let el = document.getElementById('tr-flag-indicator');
    if (el) return el;
    el = document.createElement('div');
    el.id = 'tr-flag-indicator';
    el.className = 'tr-flag-indicator';
    document.body.appendChild(el);
    return el;
  }
  function renderIndicator() {
    const frames = flaggedFrames();
    const el = ensureIndicator();
    if (!frames.length || flagCursor < 0) { el.style.display = 'none'; return; }
    const mag = flaggedMag()[flagCursor];
    el.style.display = 'flex';
    el.innerHTML =
      `<span class="tr-flag-count">jump ${flagCursor + 1}/${frames.length}</span>` +
      `<span class="tr-flag-mag">${mag ? mag.toFixed(1) + 'mm' : ''}</span>` +
      `<span class="tr-flag-time">frame ${frames[flagCursor]}</span>` +
      `<span class="tr-flag-hint">n/p</span>`;
  }
  function jumpToFlag(offset) {
    const frames = flaggedFrames();
    if (!frames.length) return false;
    if (flagCursor < 0) flagCursor = offset > 0 ? 0 : frames.length - 1;
    else flagCursor = (flagCursor + offset + frames.length) % frames.length;
    setFrame(frames[flagCursor]);
    renderIndicator();
    return true;
  }

  // ---------- Play/pause (advance frame at fps) ----------
  let playTimer = null;
  const isPlaying = () => playTimer !== null;
  function stopPlay() { if (playTimer) { clearInterval(playTimer); playTimer = null; } }
  function togglePlay() {
    if (isPlaying()) { stopPlay(); return; }
    playTimer = setInterval(() => {
      const next = currentFrame + 1;
      if (next >= nFrames) { stopPlay(); return; }
      setFrame(next);
    }, Math.max(1000 / fps, 16));
  }

  // ---------- Scrub-cache status polling ----------
  let statusTimer = null;
  async function pollStatus() {
    try {
      const r = await fetch(`/review/${sessionId}/scrub_status`);
      const s = await r.json();
      if (scrubStatus) {
        if (s.ready) {
          scrubStatus.hidden = true;
          if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
        } else {
          const pct = s.n_total > 0 ? Math.round((s.n_done / s.n_total) * 100) : 0;
          scrubStatus.hidden = false;
          scrubStatus.textContent = `· warming scrub cache ${s.n_done}/${s.n_total} (${pct}%)`;
        }
      }
    } catch (_) {}
  }

  // ---------- Keyboard ----------
  function onKey(e) {
    if (e.target.matches('input, textarea, select')) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.toLowerCase();
    const fireBtn = (key) => {
      const btn = document.querySelector(`[data-key="${key}"]`);
      if (!btn) return false;
      btn.click();
      e.preventDefault();
      return true;
    };
    if (k === 'arrowleft')  { stopPlay(); setFrame(currentFrame - (e.shiftKey ? 10 : 1)); e.preventDefault(); return; }
    if (k === 'arrowright') { stopPlay(); setFrame(currentFrame + (e.shiftKey ? 10 : 1)); e.preventDefault(); return; }
    if (k === ' ') { togglePlay(); e.preventDefault(); return; }
    if (k === 'n') { if (jumpToFlag(+1)) e.preventDefault(); return; }
    if (k === 'p') { if (jumpToFlag(-1)) e.preventDefault(); return; }
    if (k === 'a') return fireBtn('a');
    if (k === 'r') return fireBtn('r');
    if (k === 'f') return fireBtn('f');
    if (k === 'u') return fireBtn('u');
    if (k === 's') return fireBtn('s');
    if (k === ']') return fireBtn('next');
    if (k === '[') return fireBtn('prev');
  }

  window.addEventListener('DOMContentLoaded', () => {
    renderTrajectory(data);
    renderJumps(data);
    renderProprio(data);
    wireClickToFrame('tr-jumps');
    wireClickToFrame('tr-proprio');

    if (slider) {
      slider.addEventListener('input', (e) => {
        stopPlay();
        setFrame(parseInt(e.target.value, 10), { updateSlider: false });
      });
    }

    setFrame(0);
    document.addEventListener('keydown', onKey);

    // Poll cache status every 700ms until ready.
    pollStatus();
    statusTimer = setInterval(pollStatus, 700);
  });
})();
