"""Run batches of games under a policy, and the seed ranges used everywhere."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .engine import State, reset, step

# Disjoint seed ranges so nothing evaluated was ever seen in training.
TRAIN_SEED_MAX = 0xE0000000          # training draws uniformly below this
VAL_SEED_BASE = 0xE0000000           # model selection during training
TEST_SEED_BASE = 0xF0000000          # held-out boards for the reported experiment

Policy = Callable[[State], np.ndarray]


def stall_limit_for(size: int) -> int:
    """Foodless moves allowed before a game is declared stalled."""
    return 2 * size * size


def play(policy: Policy, seeds, size: int = 10, max_steps: int = 20000, stall_limit: int | None = None,
         progress: Callable[[int, State], None] | None = None) -> dict:
    """Play one game per seed to completion. The policy only ever sees live games."""
    if stall_limit is None:
        stall_limit = stall_limit_for(size)
    s = reset(seeds, size)
    steps = np.zeros(len(s), dtype=np.int32)
    t = 0
    while s.alive.any() and t < max_steps:
        idx = np.nonzero(s.alive)[0]
        actions = np.ones(len(s), dtype=np.int64)
        actions[idx] = policy(s.take(idx)) if len(idx) < len(s) else policy(s)
        step(s, actions, stall_limit)
        steps[idx] += 1
        t += 1
        if progress is not None:
            progress(t, s)
    return {"score": s.score.copy(), "steps": steps, "won": s.won.copy(), "unfinished": s.alive.copy()}
