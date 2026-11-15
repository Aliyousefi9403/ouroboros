"""The experiment: perception versus planning.

When Snake makes a better decision, how much comes from seeing more of the
board, and how much from thinking further ahead? Every agent plays the same
held-out seeds (same start, same food stream):

  reflex          11 numbers, no lookahead
  board           full board, no lookahead
  board+search    full board, B simulated moves of lookahead ranked by the network
  search-only     no learning, the same B simulated moves ranked by a distance heuristic

    python -m ouroboros.experiment --games 200
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
GAMMA = 0.97
BUDGETS = (12, 30, 100, 300, 1000)


def make_policy(kind: str, budget: int):
    from .features import obs11
    from .models import BoardLearner, ReflexLearner
    from .search import Planner, food_distance_value

    if kind == "reflex":
        net = ReflexLearner.load()
        return lambda s: net.q(obs11(s)).argmax(axis=1)
    if kind == "search-only":
        return Planner(food_distance_value(GAMMA), budget, GAMMA)
    board = BoardLearner.load()
    if budget == 0:
        return lambda s: board.q(board.encode(s)).argmax(axis=1)
    return Planner(lambda s: board.value(board.encode(s)), budget, GAMMA)


def run_config(job):
    kind, budget, games, size = job
    import torch
    torch.set_num_threads(1)
    from .play import TEST_SEED_BASE, play

    t0 = time.time()
    res = play(make_policy(kind, budget), TEST_SEED_BASE + np.arange(games), size)
    score = res["score"]
    boot = np.random.default_rng(0).choice(score, size=(4000, games)).mean(axis=1)
    return {
        "agent": kind, "budget": budget, "games": games,
        "mean": round(float(score.mean()), 2),
        "ci95": [round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)],
        "median": float(np.median(score)), "max": int(score.max()), "min": int(score.min()),
        "meanSteps": round(float(res["steps"].mean()), 1),
        "wins": int(res["won"].sum()),
        "seconds": round(time.time() - t0, 1),
        "scores": score.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--budgets", type=int, nargs="*", default=list(BUDGETS))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs = [("reflex", 0), ("board", 0)]
    jobs += [("board", b) for b in args.budgets] + [("search-only", b) for b in args.budgets]
    jobs = [(k, b, args.games, args.size) for k, b in jobs]
    jobs.sort(key=lambda j: -j[1])                       # start the slow ones first
    results = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_config, jobs):
            print({k: v for k, v in r.items() if k != "scores"}, flush=True)
            results.append(r)
    results.sort(key=lambda r: (r["agent"], r["budget"]))

    out = {"size": args.size, "games": args.games, "gamma": GAMMA, "results": results, "training": {}}
    for name in ("reflex", "board"):
        path = os.path.join(ROOT, "models", f"{name}_training.json")
        if os.path.exists(path):
            with open(path) as f:
                out["training"][name] = json.load(f)["history"]
    for path in (os.path.join(ROOT, "results", "experiment.json"), os.path.join(ROOT, "docs", "data", "experiment.json")):
        with open(path, "w") as f:
            json.dump(out, f)
    print("done")


if __name__ == "__main__":
    main()
