"""Exhaustive survival analysis: is a move provably fatal?

For each first action from a position we enumerate *every* legal continuation
with the real engine, level by level, up to ``limit`` moves. Three verdicts:

  survives  - some continuation is still alive after ``limit`` moves (proof by witness)
  doomed    - every continuation collides; ``moves`` is the longest any of them lasts
  unknown   - the frontier outgrew ``cap`` and had to be thinned before a verdict

A thinned frontier can still prove survival (a witness is a witness) but can
never prove doom, which keeps every reported "doomed" verdict exact.
"""
from __future__ import annotations

import numpy as np

from .engine import State, step

SURVIVES, DOOMED, UNKNOWN = "survives", "doomed", "unknown"


def analyse(roots: State, limit: int = 16, cap: int = 400, seed: int = 0) -> dict:
    """Analyse all three first actions of every root at once.

    Returns arrays of shape [R, 3]:
      status (0 = unknown, 1 = survives, 2 = doomed), moves (safe moves proven
      possible: ``limit`` for survivors, the exact maximum for doomed lines).
    """
    rng = np.random.default_rng(seed)
    r_n = len(roots)
    frontier = roots.repeat(3)
    group = np.arange(r_n * 3)                       # root * 3 + first action
    actions = np.tile(np.arange(3), r_n)
    step(frontier, actions)
    truncated = np.zeros(r_n * 3, dtype=bool)
    moves = np.zeros(r_n * 3, dtype=np.int32)
    won = np.zeros(r_n * 3, dtype=bool)

    won[group[frontier.won]] = True
    live = np.nonzero(frontier.alive)[0]
    frontier, group = frontier.take(live), group[live]
    moves[np.unique(group)] = 1

    for depth in range(2, limit + 1):
        if not len(frontier):
            break
        # Thin any group whose frontier is over the cap.
        counts = np.bincount(group, minlength=r_n * 3)
        if (counts > cap).any():
            order = np.lexsort((rng.random(len(group)), group))
            starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
            rank = np.arange(len(group)) - starts[group[order]]
            kept = order[rank < cap]
            truncated |= counts > cap
            frontier, group = frontier.take(kept), group[kept]
        kids = frontier.repeat(3)
        kgroup = np.repeat(group, 3)
        step(kids, np.tile(np.arange(3), len(frontier)))
        won[kgroup[kids.won]] = True
        live = np.nonzero(kids.alive)[0]
        frontier, group = kids.take(live), kgroup[live]
        moves[np.unique(group)] = depth

    survives = (moves >= limit) | won
    status = np.where(survives, 1, np.where(truncated, 0, 2))
    moves = np.where(survives, limit, moves)
    return {"status": status.reshape(r_n, 3), "moves": moves.reshape(r_n, 3)}


def verdicts(root: State, limit: int = 16, cap: int = 20000) -> list[dict]:
    """Human-readable analysis of a single position."""
    res = analyse(root, limit, cap)
    names = (UNKNOWN, SURVIVES, DOOMED)
    return [{"status": names[res["status"][0, a]], "moves": int(res["moves"][0, a])} for a in range(3)]
