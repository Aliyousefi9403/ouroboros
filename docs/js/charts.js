// Results page: hero numbers, two SVG charts and the full table.

const COLORS = { reflex: '#c98500', board: '#3987e5', 'search-only': '#d95926', 'board+search': '#199e70' };
const NAMES = { reflex: 'Reflex (11 inputs)', board: 'Full board', 'search-only': 'Search only', 'board+search': 'Full board + search' };
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

function barChart(rows, maxV) {
  const W = 520, rowH = 44, left = 150, right = 56, H = rows.length * rowH + 26;
  const x = (v) => left + (v / maxV) * (W - left - right);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Mean score by agent">`;
  for (let g = 0; g <= maxV; g += maxV > 60 ? 20 : 10) {
    s += `<line class="grid" x1="${x(g)}" x2="${x(g)}" y1="0" y2="${H - 22}"/><text x="${x(g)}" y="${H - 6}" text-anchor="middle">${g}</text>`;
  }
  rows.forEach((r, i) => {
    const y = i * rowH + 8, h = 22, w = Math.max(2, x(r.mean) - left);
    s += `<text x="${left - 10}" y="${y + h / 2 + 4}" text-anchor="end">${esc(r.label)}</text>`;
    s += `<path d="M${left} ${y}h${w - 4}a4 4 0 0 1 4 4v${h - 8}a4 4 0 0 1 -4 4h${-(w - 4)}z" fill="${r.color}"/>`;
    s += `<line x1="${x(r.ci95[0])}" x2="${x(r.ci95[1])}" y1="${y + h / 2}" y2="${y + h / 2}" stroke="#e8f1ed" stroke-width="2" stroke-linecap="round"/>`;
    s += `<text class="val" x="${x(r.ci95[1]) + 8}" y="${y + h / 2 + 4}">${r.mean.toFixed(1)}</text>`;
    s += `<rect class="hit" x="0" y="${y - 6}" width="${W}" height="${rowH - 4}" fill="transparent" data-tip="${esc(`<b>${r.label}</b><br>mean ${r.mean.toFixed(2)} (95% CI ${r.ci95[0]}–${r.ci95[1]})<br>median ${r.median} · best ${r.max} · ${r.games} games`)}"/>`;
  });
  return s + '</svg>';
}

function lineChart(series, refs, budgets, maxV) {
  const W = 520, H = 300, left = 40, right = 150, top = 10, bottom = 34;
  const lx = (b) => Math.log(b), x0 = lx(budgets[0]), x1 = lx(budgets[budgets.length - 1]);
  const x = (b) => left + ((lx(b) - x0) / (x1 - x0 || 1)) * (W - left - right);
  const y = (v) => top + (1 - v / maxV) * (H - top - bottom);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Mean score against search budget">`;
  for (let g = 0; g <= maxV; g += maxV > 60 ? 20 : 10) s += `<line class="grid" x1="${left}" x2="${W - right}" y1="${y(g)}" y2="${y(g)}"/><text x="${left - 8}" y="${y(g) + 4}" text-anchor="end">${g}</text>`;
  for (const b of budgets) s += `<text x="${x(b)}" y="${H - 14}" text-anchor="middle">${b}</text>`;
  s += `<text x="${(left + W - right) / 2}" y="${H}" text-anchor="middle">simulated moves per decision (log scale)</text>`;
  const labels = [];
  for (const r of refs) {
    s += `<line x1="${left}" x2="${W - right}" y1="${y(r.mean)}" y2="${y(r.mean)}" stroke="${r.color}" stroke-width="2" stroke-dasharray="2 6" stroke-linecap="round"/>`;
    labels.push({ y: y(r.mean), text: `${r.label} · no search`, color: r.color });
  }
  for (const sr of series) {
    const pts = sr.points.map((p) => `${x(p.budget)},${y(p.mean)}`).join(' ');
    s += `<polyline points="${pts}" fill="none" stroke="${sr.color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>`;
    for (const p of sr.points) {
      s += `<circle cx="${x(p.budget)}" cy="${y(p.mean)}" r="5" fill="${sr.color}" stroke="#101a17" stroke-width="2"/>`;
      s += `<circle cx="${x(p.budget)}" cy="${y(p.mean)}" r="16" fill="transparent" data-tip="${esc(`<b>${sr.label}</b> · budget ${p.budget}<br>mean ${p.mean.toFixed(2)} (95% CI ${p.ci95[0]}–${p.ci95[1]})`)}"/>`;
    }
    const last = sr.points[sr.points.length - 1];
    labels.push({ y: y(last.mean), text: sr.label, color: sr.color });
  }
  labels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) if (labels[i].y - labels[i - 1].y < 15) labels[i].y = labels[i - 1].y + 15;
  for (const l of labels) s += `<circle cx="${W - right + 10}" cy="${l.y}" r="4" fill="${l.color}"/><text x="${W - right + 19}" y="${l.y + 4}">${esc(l.text)}</text>`;
  return s + '</svg>';
}

export function renderExperiment(el, data, tip) {
  const get = (agent, budget) => data.results.find((r) => r.agent === agent && r.budget === budget);
  const reflex = get('reflex', 0), board = get('board', 0);
  const budgets = [...new Set(data.results.filter((r) => r.budget > 0).map((r) => r.budget))].sort((a, b) => a - b);
  const top = budgets[budgets.length - 1];
  const bs = get('board', top), so = get('search-only', top);
  const maxV = Math.ceil(Math.max(...data.results.map((r) => r.ci95[1])) / 10) * 10 + 10;
  const row = (r, key, label) => ({ ...r, color: COLORS[key], label });
  const bars = [row(reflex, 'reflex', 'Reflex · 11 inputs'), row(board, 'board', 'Full board'),
    row(so, 'search-only', `Search only · ${top}`), row(bs, 'board+search', `Board + search · ${top}`)];
  const series = [
    { label: 'Board + search', color: COLORS['board+search'], points: budgets.map((b) => get('board', b)) },
    { label: 'Search only', color: COLORS['search-only'], points: budgets.map((b) => get('search-only', b)) },
  ];
  const refs = [{ ...board, label: 'Full board', color: COLORS.board }, { ...reflex, label: 'Reflex', color: COLORS.reflex }];
  const f = (v) => (v >= 0 ? '+' : '') + v.toFixed(1);
  const rank = (r) => (r.agent === 'reflex' ? 0 : r.agent === 'search-only' ? 3 : r.budget ? 2 : 1);
  const table = [...data.results].sort((a, b) => rank(a) - rank(b) || a.budget - b.budget);

  el.innerHTML = `
  <div class="tiles">
    <div class="card tile"><b>${f(board.mean - reflex.mean)}</b><span>food per game from <em>seeing the whole board</em> (${reflex.mean.toFixed(1)} → ${board.mean.toFixed(1)}), with no lookahead at all</span></div>
    <div class="card tile"><b>${f(bs.mean - board.mean)}</b><span>more from <em>thinking ahead</em>: the same network with ${top} simulated moves per decision (${board.mean.toFixed(1)} → ${bs.mean.toFixed(1)})</span></div>
    <div class="card tile"><b>${f(bs.mean - so.mean)}</b><span>what the <em>learned value</em> adds over a hand-written heuristic at the same ${top}-move budget (${so.mean.toFixed(1)} → ${bs.mean.toFixed(1)})</span></div>
  </div>
  <div class="exp-grid">
    <div class="card chart"><h2 class="chart-title">Mean score on ${data.games} held-out games</h2><p class="chart-sub">${data.size}×${data.size} board · bars show the mean, white whiskers the 95% bootstrap interval</p>${barChart(bars, maxV)}</div>
    <div class="card chart"><h2 class="chart-title">What extra thinking buys</h2><p class="chart-sub">Mean score as the planning budget grows; dashed lines are the agents that do not search</p>${lineChart(series, refs, budgets, maxV)}</div>
  </div>
  <div class="card" style="margin-top:18px;overflow-x:auto"><h2>Every configuration</h2>
    <table class="data"><thead><tr><th>Agent</th><th>Budget</th><th>Mean</th><th>95% CI</th><th>Median</th><th>Best</th><th>Boards filled</th><th>Moves / game</th></tr></thead><tbody>
    ${table.map((r) => `<tr><td>${NAMES[r.agent === 'board' && r.budget ? 'board+search' : r.agent]}</td><td>${r.budget || '—'}</td><td>${r.mean.toFixed(2)}</td><td>${r.ci95[0]} – ${r.ci95[1]}</td><td>${r.median}</td><td>${r.max}</td><td>${r.wins}</td><td>${r.meanSteps}</td></tr>`).join('')}
    </tbody></table>
    <p class="note">Score is food eaten; ${data.size * data.size - 3} fills the board. A game ends on collision or after ${2 * data.size * data.size} moves without food. Budget counts simulated moves per decision, identical for both searching agents.</p>
  </div>`;

  el.querySelectorAll('[data-tip]').forEach((n) => {
    n.addEventListener('pointermove', (e) => {
      tip.hidden = false; tip.innerHTML = n.dataset.tip;
      tip.style.left = Math.min(e.clientX + 14, innerWidth - 270) + 'px'; tip.style.top = e.clientY + 14 + 'px';
    });
    n.addEventListener('pointerleave', () => { tip.hidden = true; });
  });
}
