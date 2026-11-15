"""Generate tests/fixtures.json: Python ground truth that the JavaScript port must reproduce.

    python scripts/make_fixtures.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ouroboros.engine import reset, step, to_json  # noqa: E402
from ouroboros.features import obs11, planes  # noqa: E402
from ouroboros.models import BoardLearner, ReflexLearner  # noqa: E402
from ouroboros.search import Planner  # noqa: E402
from ouroboros.verify import analyse  # noqa: E402

rng = np.random.default_rng(7)
reflex, board = ReflexLearner.load(), BoardLearner.load()
policy = lambda s: board.q(board.encode(s)).argmax(axis=1)

# 1. Full game traces: mostly greedy with some random moves, so games get long.
games = []
for seed in [1, 42, 2026, 0xDEADBEEF, 123456789, 0xFFFFFFFF]:
    s = reset([seed])
    actions, trace = [], []
    while s.alive[0] and len(actions) < 600:
        a = int(policy(s)[0]) if rng.random() > 0.08 else int(rng.integers(0, 3))
        step(s, [a])
        actions.append(a)
        trace.append([int(s.head[0]), int(s.food[0]), int(s.length[0]), int(s.rng[0]), int(s.alive[0])])
    games.append({"seed": seed, "actions": actions, "trace": trace, "finalTtl": s.ttl[0].tolist()})

# 2. Positions with features, network outputs, proofs and planner decisions.
s = reset(1000 + np.arange(40))
for t in range(120):
    a = policy(s)
    flip = rng.random(len(s)) < 0.05
    a[flip] = rng.integers(0, 3, size=int(flip.sum()))
    nxt = s.copy()
    step(nxt, a)
    keep = nxt.alive                    # freeze games on the move before they would die
    s.assign(np.nonzero(keep)[0], nxt.take(np.nonzero(keep)[0]))
proof = analyse(s, limit=12, cap=100000)
planner = Planner(lambda st: board.value(board.encode(st)), 60)
plan_scores = planner.scores(s)
positions = []
for i in range(len(s)):
    one = s.take([i])
    positions.append({
        "state": to_json(s, i),
        "obs11": obs11(one)[0].astype(int).tolist(),
        "planes": np.round(planes(one)[0].ravel(), 6).tolist(),
        "reflexQ": reflex.q(obs11(one))[0].tolist(),
        "boardQ": board.q(planes(one))[0].tolist(),
        "proofStatus": proof["status"][i].tolist(),
        "proofMoves": proof["moves"][i].tolist(),
        "planScores": [None if not np.isfinite(v) else float(v) for v in plan_scores[i]],
        "planAction": int(plan_scores[i].argmax()),
    })

with open(os.path.join(ROOT, "tests", "fixtures.json"), "w") as f:
    json.dump({"proofLimit": 12, "planBudget": 60, "games": games, "positions": positions}, f)
print("games", [len(g["actions"]) for g in games], "positions", len(positions),
      "proof status counts", np.bincount(proof["status"].ravel(), minlength=3).tolist())
