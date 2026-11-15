// The browser code must reproduce the Python research code exactly.
//   node --test tests/
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as E from '../docs/js/engine.js';
import { obs11, planes, parseModel, reflexQ, boardQ, plan, boardValue } from '../docs/js/agents.js';
import { analyseMove } from '../docs/js/verify.js';

const root = new URL('..', import.meta.url);
const fx = JSON.parse(readFileSync(new URL('tests/fixtures.json', root)));
const load = (name) => {
  const manifest = JSON.parse(readFileSync(new URL(`docs/models/${name}.json`, root)));
  const bin = readFileSync(new URL(`docs/models/${name}.bin`, root));
  return parseModel(manifest, bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
};
const reflex = load('reflex'), board = load('board');
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b}`);

test('engine replays Python games bit for bit', () => {
  for (const g of fx.games) {
    let st = E.reset(g.seed);
    g.actions.forEach((a, i) => {
      st = E.step(st, a);
      const [head, food, length, rng, alive] = g.trace[i];
      assert.deepEqual([st.head, st.food, st.length, st.rng, +st.alive], [head, food, length, rng, alive], `seed ${g.seed} move ${i}`);
    });
    assert.deepEqual(Array.from(st.ttl), g.finalTtl);
  }
});

test('action encoding round-trips', () => {
  for (const g of fx.games) assert.deepEqual(E.decodeActions(E.encodeActions(g.actions)), g.actions);
});

test('features match', () => {
  for (const p of fx.positions) {
    const st = E.fromJSON(p.state);
    assert.deepEqual(Array.from(obs11(st)), p.obs11);
    const pl = planes(st);
    p.planes.forEach((v, i) => close(pl[i], v, 1e-5, `plane value ${i}`));
  }
});

test('network outputs match', () => {
  for (const p of fx.positions) {
    const st = E.fromJSON(p.state);
    reflexQ(reflex, st).forEach((v, i) => close(v, p.reflexQ[i], 1e-4, 'reflex Q'));
    boardQ(board, st).forEach((v, i) => close(v, p.boardQ[i], 1e-4, 'board Q'));
  }
});

test('exhaustive proofs agree', () => {
  let proven = 0;
  for (const p of fx.positions) {
    const st = E.fromJSON(p.state);
    for (let a = 0; a < 3; a++) {
      if (p.proofStatus[a] === 0) continue;
      const v = analyseMove(st, a, fx.proofLimit, 2e6);
      assert.equal(v.status, p.proofStatus[a] === 1 ? 'survives' : 'doomed');
      assert.equal(v.moves, p.proofMoves[a]);
      if (v.status === 'survives') {       // the witness must really survive
        let s = st;
        for (const m of v.line) s = E.step(s, m);
        assert.ok(s.alive || s.won);
      }
      proven++;
    }
  }
  assert.ok(proven > 60);
});

test('planner agrees', () => {
  for (const p of fx.positions) {
    const res = plan(boardValue(board), E.fromJSON(p.state), fx.planBudget);
    res.scores.forEach((v, i) => (p.planScores[i] === null ? assert.equal(v, -Infinity) : close(v, p.planScores[i], 1e-3, 'plan score')));
    assert.equal(res.action, p.planAction);
  }
});
