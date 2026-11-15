// Canvas renderer. The snake is drawn as one continuous ribbon whose colour
// fades from head to tail, so the head-to-tail order the full-board agent
// relies on is visible to people too.

import { DX, DY, bodyCells, step } from './engine.js';

const HEAD = [255, 214, 102];   // gold
const TAIL = [16, 185, 129];    // emerald
const DEEP = [13, 84, 66];      // the tail tip, about to clear
const mix = (a, b, t) => `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(',')})`;

export function setupCanvas(canvas, cssSize) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  canvas.style.width = canvas.style.height = cssSize + 'px';
  canvas.width = canvas.height = Math.round(cssSize * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/**
 * opts: { prev, t, plan, verdicts, crash, highlight, time, mini }
 *   prev/t    previous state and 0..1 progress for motion interpolation
 *   plan      relative actions to trace from the head (the planner's intended line)
 *   verdicts  per-action { status } markers on the three neighbouring cells
 *   crash     relative action that killed the snake (draws the impact)
 */
export function draw(ctx, cssSize, st, opts = {}) {
  const size = st.size, cell = cssSize / size, mini = !!opts.mini;
  const cx = (i) => ((i % size) + 0.5) * cell, cy = (i) => (((i / size) | 0) + 0.5) * cell;
  ctx.clearRect(0, 0, cssSize, cssSize);

  // board
  ctx.fillStyle = '#0b1210';
  ctx.beginPath(); ctx.roundRect(0, 0, cssSize, cssSize, mini ? 8 : 16); ctx.fill();
  ctx.fillStyle = 'rgba(255,255,255,0.055)';
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) {
    ctx.beginPath(); ctx.arc((c + 0.5) * cell, (r + 0.5) * cell, Math.max(1, cell * 0.035), 0, 7); ctx.fill();
  }

  // food
  if (st.food >= 0) {
    const pulse = mini ? 0 : 0.5 + 0.5 * Math.sin((opts.time || 0) / 260);
    const x = cx(st.food), y = cy(st.food);
    if (!mini) {
      const g = ctx.createRadialGradient(x, y, 0, x, y, cell * 0.9);
      g.addColorStop(0, `rgba(251,113,133,${0.35 + 0.15 * pulse})`); g.addColorStop(1, 'rgba(251,113,133,0)');
      ctx.fillStyle = g; ctx.fillRect(x - cell, y - cell, cell * 2, cell * 2);
    }
    ctx.fillStyle = '#fb7185';
    ctx.beginPath(); ctx.arc(x, y, cell * (0.24 + 0.03 * pulse), 0, 7); ctx.fill();
  }

  // snake ribbon
  const cells = bodyCells(st);
  const pts = cells.map((i) => [cx(i), cy(i)]);
  const t = opts.prev && opts.t < 1 ? opts.t : 1;
  if (t < 1 && cells.length > 1) {
    const prev = opts.prev;
    pts[0] = [cx(prev.head) + (cx(st.head) - cx(prev.head)) * t, cy(prev.head) + (cy(st.head) - cy(prev.head)) * t];
    if (st.length === prev.length) {
      const prevCells = bodyCells(prev), old = prevCells[prevCells.length - 1], last = pts.length - 1;
      pts.push([cx(old) + (pts[last][0] - cx(old)) * t, cy(old) + (pts[last][1] - cy(old)) * t]);
    }
  }
  const n = pts.length;
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  for (let i = n - 1; i >= 1; i--) {
    const f = (i - 0.5) / Math.max(1, n - 1);
    ctx.strokeStyle = f < 0.5 ? mix(HEAD, TAIL, Math.pow(f * 2, 0.8)) : mix(TAIL, DEEP, (f - 0.5) * 2);
    ctx.lineWidth = cell * (0.68 - 0.24 * f);
    ctx.beginPath(); ctx.moveTo(pts[i][0], pts[i][1]); ctx.lineTo(pts[i - 1][0], pts[i - 1][1]); ctx.stroke();
  }
  // head
  const [hx, hy] = pts[0];
  ctx.fillStyle = mix(HEAD, TAIL, 0);
  ctx.beginPath(); ctx.arc(hx, hy, cell * 0.36, 0, 7); ctx.fill();
  if (!mini) {
    const fx = DX[st.dir], fy = DY[st.dir], sx = -fy, sy = fx;
    ctx.fillStyle = '#0b1210';
    for (const s of [-1, 1]) {
      ctx.beginPath();
      ctx.arc(hx + fx * cell * 0.12 + sx * s * cell * 0.15, hy + fy * cell * 0.12 + sy * s * cell * 0.15, cell * 0.055, 0, 7);
      ctx.fill();
    }
  }

  // planner's intended line
  if (opts.plan && opts.plan.length > 0 && st.alive) {
    let s = st;
    ctx.strokeStyle = 'rgba(125,211,252,0.75)'; ctx.lineWidth = Math.max(1.5, cell * 0.06);
    ctx.setLineDash([cell * 0.12, cell * 0.16]);
    ctx.beginPath(); ctx.moveTo(cx(st.head), cy(st.head));
    for (const a of opts.plan) {
      s = step(s, a);
      if (!s.alive) break;
      ctx.lineTo(cx(s.head), cy(s.head));
    }
    ctx.stroke(); ctx.setLineDash([]);
    if (s.alive) { ctx.fillStyle = 'rgba(125,211,252,0.9)'; ctx.beginPath(); ctx.arc(cx(s.head), cy(s.head), cell * 0.09, 0, 7); ctx.fill(); }
  }

  // proof markers on the three candidate cells
  if (opts.verdicts && st.alive) {
    for (let a = 0; a < 3; a++) {
      const v = opts.verdicts[a];
      if (!v) continue;
      const d = (st.dir + a - 1) & 3;
      const x = (st.head % size) + DX[d], y = ((st.head / size) | 0) + DY[d];
      if (x < 0 || x >= size || y < 0 || y >= size) continue;
      const px = (x + 0.5) * cell, py = (y + 0.5) * cell, r = cell * (mini ? 0.3 : 0.34);
      const col = v.status === 'doomed' ? '248,113,113' : v.status === 'survives' ? '74,222,128' : '148,163,184';
      ctx.strokeStyle = `rgba(${col},0.95)`; ctx.fillStyle = `rgba(${col},0.16)`; ctx.lineWidth = Math.max(1.5, cell * 0.05);
      ctx.beginPath(); ctx.roundRect(px - r, py - r, r * 2, r * 2, r * 0.35); ctx.fill(); ctx.stroke();
      if (opts.highlight === a) { ctx.lineWidth = cell * 0.1; ctx.stroke(); }
    }
  }

  // impact
  if (opts.crash != null) {
    const d = (st.dir + opts.crash - 1) & 3;
    const px = cx(st.head) + DX[d] * cell * 0.55, py = cy(st.head) + DY[d] * cell * 0.55, r = cell * 0.2;
    ctx.strokeStyle = '#f87171'; ctx.lineWidth = Math.max(2, cell * 0.09);
    ctx.beginPath(); ctx.moveTo(px - r, py - r); ctx.lineTo(px + r, py + r); ctx.moveTo(px + r, py - r); ctx.lineTo(px - r, py + r); ctx.stroke();
  }
}
