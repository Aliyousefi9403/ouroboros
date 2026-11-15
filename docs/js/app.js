import * as E from './engine.js';
import { loadModel, reflexQ, boardQ, argmax, obs11, OBS_LABELS } from './agents.js';
import { analysePosition, findFatalMove } from './verify.js';
import { setupCanvas, draw } from './render.js';
import { renderExperiment } from './charts.js';

const $ = (s) => document.querySelector(s);
const BUDGETS = [12, 30, 60, 100, 200, 300];
const GAMMA = 0.97, LIMIT = 16;
const NAMES = { human: 'You', reflex: 'Reflex', board: 'Full board', planner: 'Planner', script: 'Escape line' };
const MOVE = ['left', 'straight', 'right'];
const KEYS = { ArrowUp: 0, ArrowRight: 1, ArrowDown: 2, ArrowLeft: 3, w: 0, d: 1, s: 2, a: 3 };

let models = null;
const G = {
  states: [], actions: [], who: [], cursor: 0, seed: null,
  driver: 'planner', playing: false, fork: null, queue: [], script: null,
  lastTick: 0, animStart: 0, plan: null, planAt: -1, verdicts: null, mark: null, compare: null,
};
const cur = () => G.states[G.cursor];
const budget = () => BUDGETS[+$('#budget').value];

// ------------------------------------------------------------------ canvas
const canvas = $('#board');
let ctx, boardPx;
function resize() {
  boardPx = Math.max(240, Math.min(548, canvas.parentElement.parentElement.clientWidth - 12, innerHeight - 370));
  ctx = setupCanvas(canvas, boardPx);
}

// ------------------------------------------------------------------ game flow
function start(state, seed = null) {
  Object.assign(G, { states: [state], actions: [], who: [], cursor: 0, seed, fork: null, queue: [], script: null, plan: null, planAt: -1, mark: null, compare: null });
  $('.board-wrap').classList.remove('dead');
  story('<p class="note">Take over, play until something goes wrong, and this panel will tell you what happened.</p>');
  refresh();
}
function newSeededGame(seed) {
  start(E.reset(seed), seed);
  setDriver('planner');
  G.playing = true;
}

function decide() {
  const st = cur();
  if (G.script) {
    const a = G.script.shift();
    if (!G.script.length) G.script = null;
    return a;
  }
  if (G.driver === 'human') {
    while (G.queue.length) {
      const rel = E.relativeAction(st.dir, G.queue.shift());
      if (rel >= 0) return rel;
    }
    return 1;
  }
  if (G.driver === 'reflex') return argmax(reflexQ(models.reflex, st));
  return argmax(boardQ(models.board, st));
}

// The planner thinks in a worker; answers for positions we have left are dropped.
const worker = new Worker(new URL('./planner.worker.js', import.meta.url), { type: 'module' });
let planSeq = 0, waiting = null;
function askPlan(thenMove) {
  waiting = { id: ++planSeq, cursor: G.cursor, thenMove };
  worker.postMessage({ id: waiting.id, st: cur(), budget: budget(), gamma: GAMMA });
}
worker.onmessage = ({ data }) => {
  if (!waiting || data.id !== waiting.id) return;
  const w = waiting;
  waiting = null;
  if (w.cursor !== G.cursor) return;
  G.plan = data.plan; G.planAt = G.cursor;
  G.plan.instinct = argmax(boardQ(models.board, cur()));
  if (w.thenMove && G.playing && G.driver === 'planner' && !G.script) apply(G.plan.action, 'planner');
  else refresh();
};

function apply(action, who) {
  if (G.cursor < G.states.length - 1) {            // branching off an earlier point
    G.states.length = G.cursor + 1; G.actions.length = G.cursor; G.who.length = G.cursor;
    $('.board-wrap').classList.remove('dead');
  }
  const next = E.step(cur(), action);
  G.states.push(next); G.actions.push(action); G.who.push(who);
  G.cursor++;
  G.animStart = performance.now();
  if (!next.alive) { G.playing = false; onEnd(); }
  refresh();
}

function tick() {
  if (!cur().alive) { G.playing = false; return; }
  const who = G.script ? 'script' : G.driver;
  if (who === 'planner') {
    if (G.planAt === G.cursor && G.plan) apply(G.plan.action, 'planner');
    else if (!waiting || waiting.cursor !== G.cursor) askPlan(true);
    return;
  }
  apply(decide(), who);
  if (who === 'script' && !G.script && cur().alive) {
    G.playing = false;
    banner(`Still alive ${LIMIT} moves later. That was the way out.`);
    refresh();
  }
}

function goto(i) {
  G.cursor = Math.max(0, Math.min(G.states.length - 1, i));
  G.playing = false; G.animStart = 0; G.queue = [];
  $('.board-wrap').classList.toggle('dead', !cur().alive);
  refresh();
}

function setDriver(d) {
  G.driver = d;
  document.querySelectorAll('#drivers button').forEach((b) => b.classList.toggle('on', b.dataset.driver === d));
  $('#budgetRow').classList.toggle('off', d !== 'planner');
  $('#driverName').textContent = NAMES[d];
  refresh();
}

function takeOver() {
  if (!cur().alive) return;
  G.fork = G.cursor; G.compare = null; G.script = null; G.queue = [];
  G.playing = false;
  setDriver('human');
  banner('You have the controls. Arrow keys move one step; space runs the clock.', 5000);
}

// ------------------------------------------------------------------ end of a line
function onEnd() {
  const st = cur(), n = G.actions.length;
  $('.board-wrap').classList.add('dead');
  if (st.won) { story(`<p><strong>Board filled.</strong> A perfect game in ${n} moves.</p>`); return; }
  const lastWho = G.who[n - 1];
  let html;
  if (G.compare && lastWho !== 'human') {
    html = `<p>From the position where you took over, the <strong>${NAMES[lastWho]}</strong> lasted <strong>${n - G.compare.fork}</strong> more moves and reached a score of <strong>${E.score(st)}</strong>. You lasted ${G.compare.moves} and reached ${G.compare.score}.</p>`;
  } else if (lastWho === 'human' && G.fork != null) {
    html = `<p>You took over at move ${G.fork}, survived <strong>${n - G.fork}</strong> moves and finished on <strong>${E.score(st)}</strong>. Was it avoidable?</p>`;
  } else {
    html = `<p>The <strong>${NAMES[lastWho] || 'snake'}</strong> crashed at move ${n} with a score of <strong>${E.score(st)}</strong>.</p>`;
  }
  html += '<div class="actions"><button class="btn primary" id="why">Could it have survived?</button>';
  if (lastWho === 'human' && G.fork != null) html += '<button class="btn" id="rewindAI">Rewind and let the planner play it</button>';
  html += '</div>';
  story(html);
  $('#why').onclick = explain;
  const r = $('#rewindAI');
  if (r) r.onclick = () => {
    G.compare = { fork: G.fork, moves: G.actions.length - G.fork, score: E.score(cur()) };
    goto(G.fork);
    setDriver('planner');
    G.playing = true;
    banner('Same position, same food sequence. The planner is driving.', 4000);
  };
}

function explain() {
  story('<p class="note">Enumerating every continuation…</p>');
  setTimeout(() => {
    const res = findFatalMove(G.states, G.actions, LIMIT);
    const n = G.actions.length;
    if (!res) {
      story(`<p>No single fatal move could be proven in the last ${LIMIT} moves: the line that was played still had an escape at every point the proof could reach. The position was lost earlier than the ${LIMIT}-move horizon can show.</p>`);
      return;
    }
    const { index, played, escape } = res;
    const name = MOVE[G.actions[index]];
    if (!escape) {
      story(`<p>Inconclusive. Going back from the crash, move ${index + 1} (${name}) is the first where the search ran out of budget after ${played.nodes.toLocaleString()} positions without finding an escape or exhausting the tree. No verdict is better than a guessed one.</p>`);
      return;
    }
    G.mark = { index, action: G.actions[index], escape: escape.action };
    const after = played.moves === 0
      ? `It ran straight into ${played.cause === 'wall' ? 'the wall' : 'the body'}.`
      : `After it, <strong>every</strong> continuation collides within ${played.moves + 1} moves: all ${played.nodes.toLocaleString()} reachable positions were checked.`;
    const when = index === n - 1 ? 'The final move' : `Move ${index + 1}, ${n - index - 1} before the crash,`;
    story(`<p>${when} was the point of no return: <strong>${name}</strong>. ${after}</p>
      <p>Going <strong class="good">${MOVE[escape.action]}</strong> instead survives at least ${LIMIT} more moves. Here is a line that does it.</p>
      <div class="actions"><button class="btn primary" id="playEscape">Play the escape</button><button class="btn" id="showMoment">Show that moment</button><button class="btn" id="retry">Take over from there</button></div>`);
    const jump = () => { goto(index); };
    $('#showMoment').onclick = jump;
    $('#retry').onclick = () => { jump(); takeOver(); };
    $('#playEscape').onclick = () => { jump(); G.script = escape.line.slice(); G.playing = true; };
    jump();
  }, 30);
}

// ------------------------------------------------------------------ panels
function story(html) { $('#storyBody').innerHTML = html; }
let bannerTimer;
function banner(text, ms = 3500) {
  const b = $('#banner');
  b.textContent = text; b.hidden = false;
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => { b.hidden = true; }, ms);
}

function chip(v) {
  if (v.status === 'survives') return `<span class="chip survives">safe ≥ ${LIMIT}</span>`;
  if (v.status === 'unknown') return '<span class="chip unknown">unresolved</span>';
  return `<span class="chip doomed">${v.moves === 0 ? 'crash' : 'dead in ' + (v.moves + 1)}</span>`;
}

function refresh() {
  if (!models || !G.states.length) return;
  const st = cur();
  $('#score').textContent = E.score(st);
  $('#move').textContent = G.cursor;
  const tl = $('#timeline');
  tl.max = G.states.length - 1; tl.value = G.cursor;
  $('#playPause').textContent = G.playing ? 'Pause' : 'Play';

  // the 11 bits
  const o = obs11(st);
  $('#bits').innerHTML = OBS_LABELS.map((l, i) => `<span class="bit${o[i] ? ' on' : ''}${i < 3 ? ' danger' : ''}">${l}</span>`).join('');

  // three moves, four opinions
  const rows = { reflex: reflexQ(models.reflex, st), board: boardQ(models.board, st) };
  if (st.alive && !G.playing && G.planAt !== G.cursor && !(waiting && waiting.cursor === G.cursor)) askPlan(false);
  // While the planner drives, its row shows the decision it has just made.
  const justMade = G.playing && G.driver === 'planner' && G.planAt === G.cursor - 1;
  const showPlan = G.plan && st.alive && (G.planAt === G.cursor || justMade);
  rows.planner = showPlan ? G.plan.scores : null;
  G.verdicts = st.alive ? analysePosition(st, LIMIT, G.playing ? 3000 : 150000) : null;
  for (const tr of document.querySelectorAll('#mind tbody tr')) {
    const key = tr.dataset.row, cells = tr.querySelectorAll('td');
    tr.classList.toggle('active', key === G.driver);
    if (key === 'proof') {
      cells.forEach((c, a) => { c.innerHTML = G.verdicts ? chip(G.verdicts[a]) : '—'; c.className = ''; });
      continue;
    }
    const q = rows[key], best = q ? argmax(q) : -1;
    cells.forEach((c, a) => {
      c.textContent = q ? (Number.isFinite(q[a]) ? q[a].toFixed(2) : '−∞') : '·';
      c.className = a === best ? 'pick' : '';
    });
  }
  const note = $('#mindNote');
  if (rows.planner && st.alive) {
    const changed = G.plan.instinct !== G.plan.action;
    note.innerHTML = `Planner looked ${G.plan.depth} moves deep using ${G.plan.nodes} simulated moves${changed ? ` and <strong style="color:var(--sky)">overruled</strong> the network's first instinct (${MOVE[G.plan.instinct]} → ${MOVE[G.plan.action]}).` : ' and agreed with the network\'s first instinct.'}`;
  } else {
    note.textContent = 'The proof row enumerates every continuation of each move with the real game engine.';
  }
}

function frame() {
  requestAnimationFrame(frame);
  const now = performance.now();
  if (!models || !G.states.length) return;
  const speed = +$('#speed').value;
  const interval = 1000 / (G.driver === 'human' && !G.script ? Math.min(speed, 7) : speed);
  if (G.playing && now - G.lastTick >= interval && !(waiting && waiting.thenMove && waiting.cursor === G.cursor)) { G.lastTick = now; tick(); }
  const st = cur(), dead = !st.alive;
  const prev = G.cursor > 0 && !dead ? G.states[G.cursor - 1] : null;
  const t = G.animStart ? Math.min(1, (now - G.animStart) / Math.min(interval, 140)) : 1;
  const marked = G.mark && G.mark.index === G.cursor;
  draw(ctx, boardPx, st, {
    prev, t, time: now,
    plan: !G.plan ? null : G.planAt === G.cursor ? G.plan.line
      : G.playing && G.driver === 'planner' && G.planAt === G.cursor - 1 ? G.plan.line.slice(1) : null,
    verdicts: G.verdicts,
    highlight: marked ? G.mark.action : null,
    crash: dead && !st.won && G.cursor > 0 ? G.actions[G.cursor - 1] : null,
  });
}

// ------------------------------------------------------------------ input
addEventListener('keydown', (e) => {
  if (!$('#arena').classList.contains('active') || e.metaKey || e.ctrlKey || e.altKey) return;
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (key === ' ') { e.preventDefault(); if (cur().alive) { G.playing = !G.playing; refresh(); } return; }
  if (key === ',') return goto(G.cursor - 1);
  if (key === '.') return goto(G.cursor + 1);
  if (!(key in KEYS)) return;
  e.preventDefault();
  if (!cur().alive) return;
  if (G.driver !== 'human' || G.script) { takeOver(); return; }
  if (G.fork == null || G.cursor < G.fork) G.fork = G.cursor;
  if (G.playing) { if (G.queue.length < 2) G.queue.push(KEYS[key]); return; }
  const rel = E.relativeAction(cur().dir, KEYS[key]);
  if (rel >= 0) apply(rel, 'human');
});
$('#playPause').onclick = () => { if (cur().alive) { G.playing = !G.playing; refresh(); } };
$('#back').onclick = () => goto(G.cursor - 1);
$('#fwd').onclick = () => goto(G.cursor + 1);
$('#timeline').oninput = (e) => goto(+e.target.value);
$('#newGame').onclick = () => { newSeededGame((Math.random() * 0xD0000000) >>> 0); };
$('#budget').oninput = () => { $('#budgetVal').textContent = budget(); G.planAt = -1; waiting = null; refresh(); };
document.querySelectorAll('#drivers button').forEach((b) => {
  b.onclick = () => (b.dataset.driver === 'human' ? takeOver() : (setDriver(b.dataset.driver), G.script = null));
});
$('#share').onclick = async () => {
  const hash = G.seed != null
    ? `s=${G.seed}&a=${E.encodeActions(G.actions.slice(0, G.cursor))}`
    : `p=${btoa(JSON.stringify(E.toJSON(cur())))}`;
  const url = location.origin + location.pathname + '#' + hash;
  try { await navigator.clipboard.writeText(url); banner('Challenge link copied. Whoever opens it gets this exact position.'); }
  catch { history.replaceState(null, '', '#' + hash); banner('Link is in the address bar.'); }
};
addEventListener('resize', resize);

function loadFromHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  try {
    if (h.has('p')) { openPosition(E.fromJSON(JSON.parse(atob(h.get('p')))), 'Challenge loaded. Your move.'); return true; }
    if (h.has('s')) {
      const seed = +h.get('s') >>> 0;
      start(E.reset(seed), seed);
      for (const a of E.decodeActions(h.get('a') || '0.')) {
        if (!cur().alive) break;
        G.states.push(E.step(cur(), a)); G.actions.push(a); G.who.push('planner'); G.cursor++;
      }
      setDriver('human'); G.fork = G.cursor; G.playing = false;
      banner('Challenge loaded. Your move.', 6000);
      refresh();
      return true;
    }
  } catch (err) { console.warn('bad challenge link', err); }
  return false;
}

function openPosition(state, message) {
  start(state, null);
  setDriver('human'); G.fork = 0; G.playing = false;
  showTab('arena');
  banner(message, 6000);
  refresh();
}

// ------------------------------------------------------------------ tabs
function showTab(name) {
  document.querySelectorAll('.view').forEach((v) => v.classList.toggle('active', v.id === name));
  document.querySelectorAll('.tabs button').forEach((b) => b.setAttribute('aria-selected', b.dataset.tab === name));
  if (name !== 'arena') G.playing = false;
  if (name === 'arena') { resize(); refresh(); }
  scrollTo({ top: 0 });
}
document.querySelectorAll('.tabs button').forEach((b) => { b.onclick = () => showTab(b.dataset.tab); });

// ------------------------------------------------------------------ puzzles
function mini(state, px, verdicts) {
  const c = document.createElement('canvas');
  draw(setupCanvas(c, px), px, state, { mini: true, verdicts });
  return c;
}
const verdictOf = (e) => e.status.map((s, a) => ({ status: s === 1 ? 'survives' : 'doomed', moves: e.moves[a] }));
const safeMoves = (e) => e.status.map((s, a) => (s === 1 ? MOVE[a] : null)).filter(Boolean).join(' or ');

function buildPuzzles(data) {
  $('#pzLimit').textContent = data.limit;
  let solved = 0, tried = 0;
  const tally = () => { $('#pzScore').textContent = tried ? `You: ${solved} / ${tried} pairs survived on both boards · an 11-input agent: 0 / ${tried}, by construction` : ''; };
  for (const pair of data.pairs.slice(0, 8)) {
    const el = document.createElement('div');
    el.className = 'card pair';
    const picks = {};
    const sides = ['a', 'b'].map((k) => {
      const wrap = document.createElement('div');
      wrap.className = 'pz-board';
      const st = E.fromJSON(pair[k].state);
      wrap.innerHTML = `<span class="lbl">Board ${k.toUpperCase()}</span>`;
      let cv = mini(st, 230);
      wrap.append(cv);
      const row = document.createElement('div');
      row.className = 'choices';
      MOVE.forEach((m, a) => {
        const b = document.createElement('button');
        b.textContent = m;
        b.onclick = () => { picks[k] = a; row.querySelectorAll('button').forEach((x) => x.classList.toggle('sel', x === b)); if ('a' in picks && 'b' in picks) reveal(); };
        row.append(b);
      });
      wrap.append(row);
      return { k, wrap, row, st, show() { const n = mini(st, 230, verdictOf(pair[k])); cv.replaceWith(n); cv = n; } };
    });
    const explain = document.createElement('div');
    explain.className = 'explain';
    explain.innerHTML = `<div class="bits">${OBS_LABELS.map((l, i) => (pair.obs[i] ? `<span class="bit on${i < 3 ? ' danger' : ''}">${l}</span>` : '')).join('')}</div>
      <p>That is everything the reflex agent knows about <em>both</em> boards. Which way would you go on each?</p>`;
    function reveal() {
      let ok = true;
      for (const s of sides) {
        s.show();
        s.row.querySelectorAll('button').forEach((b, a) => {
          b.disabled = true;
          if (pair[s.k].status[a] === 1) b.classList.add('right');
          else if (picks[s.k] === a) b.classList.add('wrong');
        });
        if (pair[s.k].status[picks[s.k]] !== 1) ok = false;
      }
      tried++; if (ok) solved++; tally();
      const r = pair.a.reflex;
      const fate = (e) => (e.status[r] === 1 ? '<span class="good">survives</span>' : `<span class="bad">${e.moves[r] === 0 ? 'crashes at once' : 'is dead within ' + (e.moves[r] + 1) + ' moves'}</span>`);
      explain.innerHTML += `<p>${ok ? '<strong class="good">You survived both.</strong>' : '<strong class="bad">One of those was fatal.</strong>'} Board A needs <strong>${safeMoves(pair.a)}</strong>; board B needs <strong>${safeMoves(pair.b)}</strong>. No move works on both.</p>
        <p>The reflex agent gets identical inputs, so it must answer identically: it goes <strong>${MOVE[r]}</strong>. On A it ${fate(pair.a)}; on B it ${fate(pair.b)}.</p>
        <div class="choices"><button data-open="a">Play board A</button><button data-open="b">Play board B</button></div>`;
      explain.querySelectorAll('[data-open]').forEach((b) => { b.onclick = () => openPosition(E.fromJSON(pair[b.dataset.open].state), 'Puzzle position. Your move.'); });
    }
    el.append(sides[0].wrap, sides[1].wrap, explain);
    $('#pairs').append(el);
  }
  for (const trap of data.traps.slice(0, 12)) {
    const st = E.fromJSON(trap.state);
    const el = document.createElement('button');
    el.className = 'card trap';
    el.append(mini(st, 170));
    const p = document.createElement('p');
    p.innerHTML = `Reflex goes <strong>${MOVE[trap.reflex]}</strong>: <span class="bad">dead in ${trap.moves[trap.reflex] + 1}</span>. Food is ${trap.foodDist} away.`;
    el.append(p);
    el.onclick = () => openPosition(st, 'The reflex agent dies here. Can you find the way out?');
    $('#traps').append(el);
  }
}

// ------------------------------------------------------------------ boot
async function boot() {
  resize();
  const [reflex, board] = await Promise.all([loadModel('models', 'reflex'), loadModel('models', 'board')]);
  models = { reflex, board };
  $('#budgetVal').textContent = budget();
  if (!loadFromHash()) newSeededGame(20260);
  requestAnimationFrame(frame);
  fetch('data/puzzles.json').then((r) => r.json()).then(buildPuzzles).catch(() => { $('#pairs').innerHTML = '<p class="note">Puzzle data not found.</p>'; });
  fetch('data/experiment.json').then((r) => r.json()).then((d) => renderExperiment($('#expBody'), d, $('#tip'))).catch(() => { $('#expBody').innerHTML = '<p class="note">Run <code>python -m ouroboros.experiment</code> to generate results.</p>'; });
  window.__ouroboros = { G, goto, takeOver, setDriver, showTab };
}
boot();
