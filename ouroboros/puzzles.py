"""Mine positions that expose what the 11-input agent cannot see.

Two kinds of puzzle are produced, both backed by exhaustive proofs:

traps  - the 11-input agent's chosen move is provably fatal within the horizon
         while another move provably survives it.
pairs  - two positions with *identical* 11-number observations whose sets of
         surviving first moves are disjoint. Any policy that only sees those
         11 numbers must lose at least one of the two boards.

    python -m ouroboros.puzzles
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .engine import State, concat, reset, step, to_json
from .features import danger, obs11, obs_key
from .models import BoardLearner, ReflexLearner
from .play import TRAIN_SEED_MAX, stall_limit_for
from .verify import analyse

PUZZLE_SEED_BASE = 0xD0000000   # inside the training range's top end, never used for evaluation


def collect(policy, seeds, eps: float, min_length: int, rng: np.random.Generator, size: int = 10) -> State:
    """Play games and keep every position with a long enough snake."""
    s = reset(seeds, size)
    kept: list[State] = []
    stall = stall_limit_for(size)
    while s.alive.any():
        idx = np.nonzero(s.alive & (s.length >= min_length))[0]
        if idx.size:
            kept.append(s.take(idx))
        a = policy(s)
        explore = rng.random(len(s)) < eps
        a[explore] = rng.integers(0, 3, size=int(explore.sum()))
        step(s, a, stall)
    return concat(kept)


def mine(n_games: int = 400, limit: int = 16, cap: int = 80, chunk: int = 400, max_positions: int = 16000, seed: int = 0, log=print) -> dict:
    rng = np.random.default_rng(seed)
    reflex, board = ReflexLearner.load(), BoardLearner.load()
    reflex_policy = lambda s: reflex.q(obs11(s)).argmax(axis=1)
    board_policy = lambda s: board.q(board.encode(s)).argmax(axis=1)
    seeds = PUZZLE_SEED_BASE + np.arange(2 * n_games)
    pos = concat([
        collect(reflex_policy, seeds[:n_games], 0.03, 8, rng),
        collect(board_policy, seeds[n_games:], 0.03, 8, rng),
    ])
    # Thin to a manageable, decorrelated sample.
    keep = rng.permutation(len(pos))[:max_positions]
    pos = pos.take(keep)
    log(f"analysing {len(pos)} positions")

    status = np.zeros((len(pos), 3), dtype=np.int64)
    moves = np.zeros((len(pos), 3), dtype=np.int64)
    for i in range(0, len(pos), chunk):
        res = analyse(pos.take(np.arange(i, min(i + chunk, len(pos)))), limit, cap, seed=i)
        status[i:i + chunk], moves[i:i + chunk] = res["status"], res["moves"]
    proven = (status != 0).all(axis=1)
    safe = status == 1
    obs = obs11(pos)
    key = obs_key(obs)
    dg = danger(pos)
    choice = reflex.q(obs).argmax(axis=1)
    rows = np.arange(len(pos))
    size = pos.size
    food_dist = np.abs(pos.head % size - pos.food % size) + np.abs(pos.head // size - pos.food // size)

    def entry(i: int) -> dict:
        return {"state": to_json(pos, int(i)), "status": status[i].tolist(), "moves": moves[i].tolist(),
                "reflex": int(choice[i]), "foodDist": int(food_dist[i])}

    # --- traps -------------------------------------------------------------
    is_trap = proven & safe.any(axis=1) & (status[rows, choice] == 2) & ~dg[rows, choice]
    depth = moves[rows, choice]
    trap_idx = np.nonzero(is_trap & (depth >= 4))[0]
    # Prefer food close by (tempting) and a slow death (subtle).
    order = np.lexsort((-depth[trap_idx], food_dist[trap_idx] > 3))
    traps, seen = [], set()
    for i in trap_idx[order]:
        sig = (int(pos.rng[i]), int(pos.length[i]) // 3)       # avoid near-duplicates from one game
        if sig in seen:
            continue
        seen.add(sig)
        traps.append(entry(i))
        if len(traps) == 24:
            break

    # --- two-board puzzles ---------------------------------------------------
    mask = (safe * np.array([1, 2, 4])).sum(axis=1)
    pairs = []
    for k in np.unique(key[proven]):
        members = np.nonzero(proven & (key == k) & (mask > 0))[0]
        best = None
        for m1 in np.unique(mask[members]):
            for m2 in np.unique(mask[members]):
                if m1 >= m2 or (m1 & m2):
                    continue
                a_set, b_set = members[mask[members] == m1], members[mask[members] == m2]
                # Quality: how long the *other* board's winning move lasts here before it dies.
                def lure(i, other_mask):
                    acts = [a for a in range(3) if other_mask >> a & 1]
                    return min(moves[i, a] for a in acts)
                a = max(a_set, key=lambda i: lure(i, m2))
                b = max(b_set, key=lambda i: lure(i, m1))
                q = min(lure(a, m2), lure(b, m1))
                if best is None or q > best[0]:
                    best = (q, a, b)
        if best:
            q, a, b = best
            pairs.append({"quality": int(q), "obs": obs[a].astype(int).tolist(), "a": entry(a), "b": entry(b)})
    pairs.sort(key=lambda p: -p["quality"])
    log(f"{int(is_trap.sum())} traps, {len(pairs)} observation classes with a two-board puzzle")
    return {
        "limit": limit, "positions": int(len(pos)), "proven": int(proven.sum()),
        "trapCount": int(is_trap.sum()), "pairClasses": len(pairs),
        "traps": traps, "pairs": pairs[:16],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "data", "puzzles.json"))
    args = ap.parse_args()
    out = mine(args.games)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
