// Feature encodings, network inference and the budgeted planner, in plain
// JavaScript. Each function mirrors its Python counterpart (features.py,
// models.py, search.py) and is checked against it in tests/parity.test.mjs.

import { DX, DY, step } from './engine.js';

// ---- the 11-number observation -------------------------------------------
export function danger(st) {
  const size = st.size, out = [false, false, false];
  for (let a = 0; a < 3; a++) {
    const d = (st.dir + a - 1) & 3;
    const nx = (st.head % size) + DX[d], ny = ((st.head / size) | 0) + DY[d];
    out[a] = nx < 0 || nx >= size || ny < 0 || ny >= size || st.ttl[ny * size + nx] > 1;
  }
  return out;
}

export const OBS_LABELS = [
  'danger ahead', 'danger right', 'danger left',
  'heading left', 'heading right', 'heading up', 'heading down',
  'food left', 'food right', 'food up', 'food down',
];

export function obs11(st) {
  const size = st.size, dg = danger(st);
  const hx = st.head % size, hy = (st.head / size) | 0;
  const has = st.food >= 0, fx = st.food % size, fy = (st.food / size) | 0;
  return Float32Array.from([
    dg[1], dg[2], dg[0],
    st.dir === 3, st.dir === 1, st.dir === 0, st.dir === 2,
    has && fx < hx, has && fx > hx, has && fy < hy, has && fy > hy,
  ].map(Number));
}

// ---- the full egocentric board -------------------------------------------
export const N_PLANES = 5;

/** Float32Array [5, P, P] (P = size + 2): wall, occupied, time-to-clear, head, food,
 *  rotated so the snake always heads up. */
export function planes(st, out) {
  const size = st.size, p = size + 2, pp = p * p;
  out = out || new Float32Array(N_PLANES * pp);
  out.fill(0);
  const inv = 1 / (size * size), dir = st.dir;
  // Where does source cell (r, c) of the padded board land after rotation?
  const dest = (r, c) =>
    dir === 0 ? r * p + c :
    dir === 1 ? (p - 1 - c) * p + r :
    dir === 2 ? (p - 1 - r) * p + (p - 1 - c) :
    c * p + (p - 1 - r);
  for (let i = 0; i < p; i++) {
    out[dest(0, i)] = out[dest(p - 1, i)] = out[dest(i, 0)] = out[dest(i, p - 1)] = 1;
  }
  for (let i = 0; i < size * size; i++) {
    const t = st.ttl[i];
    if (t > 0) {
      const j = dest(((i / size) | 0) + 1, (i % size) + 1);
      out[pp + j] = 1;
      out[2 * pp + j] = t * inv;
    }
  }
  out[3 * pp + dest(((st.head / size) | 0) + 1, (st.head % size) + 1)] = 1;
  if (st.food >= 0) out[4 * pp + dest(((st.food / size) | 0) + 1, (st.food % size) + 1)] = 1;
  return out;
}

// ---- model loading --------------------------------------------------------
export function parseModel(manifest, buffer) {
  const all = new Float32Array(buffer);
  const t = {};
  for (const { name, shape, offset } of manifest.tensors) {
    t[name] = { shape, data: all.subarray(offset, offset + shape.reduce((a, b) => a * b, 1)) };
  }
  return { ...manifest, t };
}

export async function loadModel(base, name) {
  const [manifest, buffer] = await Promise.all([
    fetch(`${base}/${name}.json`).then((r) => r.json()),
    fetch(`${base}/${name}.bin`).then((r) => r.arrayBuffer()),
  ]);
  return parseModel(manifest, buffer);
}

// ---- 11-input MLP (weights stored [in, out]) --------------------------------
export function reflexQ(model, st) {
  let h = obs11(st);
  const n = Object.keys(model.t).length / 2;
  for (let l = 0; l < n; l++) {
    const w = model.t['p' + 2 * l], b = model.t['p' + (2 * l + 1)];
    const [nin, nout] = w.shape;
    const o = new Float32Array(nout);
    for (let j = 0; j < nout; j++) o[j] = b.data[j];
    for (let i = 0; i < nin; i++) {
      const x = h[i];
      if (x === 0) continue;
      const row = i * nout;
      for (let j = 0; j < nout; j++) o[j] += x * w.data[row + j];
    }
    if (l < n - 1) for (let j = 0; j < nout; j++) if (o[j] < 0) o[j] = 0;
    h = o;
  }
  return h;
}

// ---- board CNN (PyTorch layouts: conv [O, C, 3, 3], linear [O, I]) ----------
function conv3x3relu(x, c, h, w, wt, bias, pad) {
  const o = wt.shape[0], oh = h + 2 * pad - 2, ow = w + 2 * pad - 2;
  const out = new Float32Array(o * oh * ow);
  const wd = wt.data;
  for (let oc = 0; oc < o; oc++) {
    const base = oc * oh * ow;
    out.fill(bias.data[oc], base, base + oh * ow);
    for (let ic = 0; ic < c; ic++) {
      const xin = ic * h * w, wbase = (oc * c + ic) * 9;
      for (let ky = 0; ky < 3; ky++) {
        for (let kx = 0; kx < 3; kx++) {
          const k = wd[wbase + ky * 3 + kx];
          const y0 = Math.max(0, pad - ky), y1 = Math.min(oh, h + pad - ky);
          const x0 = Math.max(0, pad - kx), x1 = Math.min(ow, w + pad - kx);
          for (let y = y0; y < y1; y++) {
            let src = xin + (y + ky - pad) * w + (x0 + kx - pad);
            let dst = base + y * ow + x0;
            for (let xx = x0; xx < x1; xx++) out[dst++] += k * x[src++];
          }
        }
      }
    }
  }
  for (let i = 0; i < out.length; i++) if (out[i] < 0) out[i] = 0;
  return [out, o, oh, ow];
}

function linear(x, wt, bias, relu) {
  const [nout, nin] = wt.shape, out = new Float32Array(nout), wd = wt.data;
  for (let j = 0; j < nout; j++) {
    let s = bias.data[j];
    const row = j * nin;
    for (let i = 0; i < nin; i++) s += wd[row + i] * x[i];
    out[j] = relu && s < 0 ? 0 : s;
  }
  return out;
}

const scratch = new Map();
export function boardQ(model, st) {
  const p = st.size + 2;
  let buf = scratch.get(p);
  if (!buf) scratch.set(p, (buf = new Float32Array(N_PLANES * p * p)));
  const t = model.t;
  let [x, c, h, w] = conv3x3relu(planes(st, buf), N_PLANES, p, p, t['0_weight'], t['0_bias'], 1);
  [x, c, h, w] = conv3x3relu(x, c, h, w, t['2_weight'], t['2_bias'], 0);
  [x, c, h, w] = conv3x3relu(x, c, h, w, t['4_weight'], t['4_bias'], 0);
  return linear(linear(x, t['7_weight'], t['7_bias'], true), t['9_weight'], t['9_bias'], false);
}

export const argmax = (q) => (q[0] >= q[1] && q[0] >= q[2] ? 0 : q[1] >= q[2] ? 1 : 2);
export const boardValue = (model) => (st) => Math.max(...boardQ(model, st));

export function foodDistanceValue(gamma) {
  return (st) => {
    if (st.food < 0) return 0;
    const size = st.size;
    const dist = Math.abs((st.head % size) - (st.food % size)) + Math.abs(((st.head / size) | 0) - ((st.food / size) | 0));
    return Math.pow(gamma, dist);
  };
}

// ---- budgeted beam search (mirrors search.py) --------------------------------
export function beamShape(budget) {
  const width = Math.max(1, Math.round(Math.sqrt(budget / 3)));
  return [width, Math.max(1, Math.floor(budget / (3 * width)))];
}

/** Returns { scores: [3], action, nodes, depth, line } where `line` is the action
 *  sequence leading to the best leaf (for drawing the plan). */
export function plan(valueFn, st, budget, gamma = 0.97) {
  const [width, depth] = beamShape(budget);
  const best = [-Infinity, -Infinity, -Infinity];
  const bestLine = [null, null, null];
  const record = (first, s, line) => { if (s > best[first]) { best[first] = s; bestLine[first] = line; } };
  let beam = [{ st, ret: 0, first: -1, line: [] }];
  let nodes = 0, reached = 0;
  for (let d = 0; d < depth && beam.length; d++) {
    const kids = [];
    for (const node of beam) {
      for (let a = 0; a < 3; a++) {
        const next = step(node.st, a);
        nodes++;
        const first = d === 0 ? a : node.first;
        const line = node.line.concat(a);
        if (!next.alive) {
          record(first, node.ret + Math.pow(gamma, d) * (next.won ? 1 : -1), line);
        } else {
          const ret = node.ret + (next.ate ? Math.pow(gamma, d) : 0);
          kids.push({ st: next, ret, first, line, score: ret + Math.pow(gamma, d + 1) * valueFn(next) });
        }
      }
    }
    reached = d + 1;
    kids.sort((a, b) => b.score - a.score);
    const cut = d === depth - 1 ? 0 : width;
    for (let i = cut; i < kids.length; i++) record(kids[i].first, kids[i].score, kids[i].line);
    beam = kids.slice(0, cut);
  }
  const action = argmax(best);
  return { scores: best, action, nodes, depth: reached, line: bestLine[action] || [action] };
}
