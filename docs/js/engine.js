// Deterministic Snake engine. Mirrors ouroboros/engine.py bit for bit: the same
// time-to-live board, the same mulberry32 food stream, the same collision rule.
// tests/parity.test.mjs replays Python-generated games through this file.

export const DX = [0, 1, 0, -1];
export const DY = [-1, 0, 1, 0];
export const LEFT = 0, STRAIGHT = 1, RIGHT = 2;
export const ACTION_NAMES = ['left', 'straight', 'right'];

function rngNext(s) {
  s = (s + 0x6D2B79F5) >>> 0;
  let t = Math.imul(s ^ (s >>> 15), s | 1);
  t = (t + Math.imul(t ^ (t >>> 7), t | 61)) ^ t;
  return [s, (t ^ (t >>> 14)) >>> 0];
}

function placeFood(st) {
  const [rng, r] = rngNext(st.rng);
  st.rng = rng;
  const ttl = st.ttl;
  let n = 0;
  for (let i = 0; i < ttl.length; i++) if (ttl[i] === 0) n++;
  if (n === 0) { st.food = -1; st.won = true; st.alive = false; return; }
  let k = r % n;
  for (let i = 0; i < ttl.length; i++) {
    if (ttl[i] === 0 && k-- === 0) { st.food = i; return; }
  }
}

export function reset(seed, size = 10) {
  const ttl = new Int16Array(size * size);
  const head = (size >> 1) * size + (size >> 1);
  ttl[head] = 3; ttl[head - 1] = 2; ttl[head - 2] = 1;
  const st = { size, ttl, head, dir: 1, length: 3, food: -1, rng: seed >>> 0, alive: true, won: false, since: 0 };
  placeFood(st);
  return st;
}

export function clone(st) {
  return { ...st, ttl: st.ttl.slice() };
}

/** Returns the successor state (never mutates). On death the board is left as it was. */
export function step(st, action) {
  if (!st.alive) return st;
  const size = st.size;
  const d = (st.dir + action - 1) & 3;
  const nx = (st.head % size) + DX[d];
  const ny = ((st.head / size) | 0) + DY[d];
  if (nx < 0 || nx >= size || ny < 0 || ny >= size) return { ...st, alive: false, cause: 'wall' };
  const cell = ny * size + nx;
  const ate = cell === st.food;
  // The tail moves first (unless growing), so its cell (ttl 1) is enterable.
  if (!ate && st.ttl[cell] > 1) return { ...st, alive: false, cause: 'body' };
  const ttl = st.ttl.slice();
  if (!ate) for (let i = 0; i < ttl.length; i++) if (ttl[i] > 0) ttl[i]--;
  const next = { ...st, ttl, head: cell, dir: d, length: st.length + (ate ? 1 : 0), since: ate ? 0 : st.since + 1, ate };
  ttl[cell] = next.length;
  if (ate) placeFood(next);
  return next;
}

export function score(st) { return st.length - 3; }

/** Cells of the snake, head first. */
export function bodyCells(st) {
  const cells = [];
  for (let i = 0; i < st.ttl.length; i++) if (st.ttl[i] > 0) cells.push(i);
  cells.sort((a, b) => st.ttl[b] - st.ttl[a]);
  return cells;
}

/** Relative action that turns heading `dir` into absolute direction `abs`, or -1 for a reversal. */
export function relativeAction(dir, abs) {
  const diff = (abs - dir + 4) & 3;
  return diff === 0 ? STRAIGHT : diff === 1 ? RIGHT : diff === 3 ? LEFT : -1;
}

// ---- serialisation -------------------------------------------------------
export function toJSON(st) {
  return { size: st.size, ttl: Array.from(st.ttl), head: st.head, dir: st.dir, length: st.length, food: st.food, rng: st.rng };
}
export function fromJSON(o) {
  return { size: o.size, ttl: Int16Array.from(o.ttl), head: o.head, dir: o.dir, length: o.length, food: o.food, rng: o.rng >>> 0, alive: true, won: false, since: 0 };
}

const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_';
/** Pack relative actions three per character (3^3 = 27 < 64). */
export function encodeActions(actions) {
  let out = '';
  for (let i = 0; i < actions.length; i += 3) {
    const a = actions[i], b = actions[i + 1] ?? 1, c = actions[i + 2] ?? 1;
    out += B64[a * 9 + b * 3 + c];
  }
  return actions.length + '.' + out;
}
export function decodeActions(str) {
  const dot = str.indexOf('.');
  const n = parseInt(str.slice(0, dot), 10);
  const out = [];
  for (const ch of str.slice(dot + 1)) {
    const v = B64.indexOf(ch);
    out.push((v / 9) | 0, ((v / 3) | 0) % 3, v % 3);
  }
  return out.slice(0, n);
}
