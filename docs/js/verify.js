// Exhaustive survival analysis in the browser (depth-first counterpart of
// ouroboros/verify.py, cross-checked against it in tests/parity.test.mjs).
//
// For one first move, enumerate every legal continuation with the real engine:
//   survives - a line that is still alive after `limit` moves exists (returned as a witness)
//   doomed   - every line collides; `moves` is the longest any of them lasts
//   unknown  - the node budget ran out before either could be established

import { step } from './engine.js';

export function analyseMove(st, action, limit = 16, nodeCap = 300000) {
  let nodes = 0, bestDepth = 0, bestLine = [];
  const line = [];
  let exhausted = false;

  // Returns true as soon as a line surviving to `limit` is found.
  function dfs(state, depth) {
    if (depth > bestDepth) { bestDepth = depth; bestLine = line.slice(); }
    if (depth >= limit || state.won) return true;
    // Straight first: witnesses are usually found without turning much.
    for (const a of [1, 0, 2]) {
      if (nodes >= nodeCap) { exhausted = true; return false; }
      nodes++;
      const next = step(state, a);
      if (!next.alive && !next.won) continue;
      line.push(a);
      if (dfs(next, depth + 1)) return true;
      line.pop();
    }
    return false;
  }

  nodes++;
  const first = step(st, action);
  if (!first.alive && !first.won) return { status: 'doomed', moves: 0, line: [action], nodes, cause: first.cause };
  line.push(action);
  const ok = dfs(first, 1);
  const status = ok ? 'survives' : exhausted ? 'unknown' : 'doomed';
  return { status, moves: ok ? limit : bestDepth, line: bestLine, nodes };
}

export function analysePosition(st, limit = 16, nodeCap = 300000) {
  return [0, 1, 2].map((a) => analyseMove(st, a, limit, nodeCap));
}

/** Walk a played line backwards to find the point of no return: the last move
 *  before which survival to `limit` was still possible and after which it was not.
 *  `states[i]` is the position before `actions[i]`. */
export function findFatalMove(states, actions, limit = 16, nodeCap = 300000) {
  for (let i = actions.length - 1; i >= 0 && i >= actions.length - limit - 1; i--) {
    const played = analyseMove(states[i], actions[i], limit, nodeCap);
    if (played.status === 'survives') return null;           // nothing provable this far back
    const others = [0, 1, 2].filter((a) => a !== actions[i]).map((a) => ({ action: a, ...analyseMove(states[i], a, limit, nodeCap) }));
    const escape = others.find((o) => o.status === 'survives');
    if (escape) return { index: i, played, escape, others };
    if (played.status === 'unknown') return { index: i, played, escape: null, others };
  }
  return null;
}
