"""Budgeted lookahead: beam search over simulated futures.

The planner is given a hard budget of simulated moves per decision. It expands
the game tree level by level with the real engine (including the food stream),
scores every frontier position as

    discounted reward collected so far  +  gamma^depth * V(position)

and keeps the best ``width`` positions per level. An action's value is the
maximum over every leaf beneath it: finished lines, positions that dropped out
of the beam, and the final frontier. ``V`` is pluggable: the
full-board network's own value estimate, or a hand-written heuristic for the
search-only baseline. Because everything is batched, one call plans for many
games at once.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .engine import State, step

ValueFn = Callable[[State], np.ndarray]


def beam_shape(budget: int) -> tuple[int, int]:
    """(width, depth) so that 3 * width * depth <= budget, roughly square."""
    width = max(1, int(round((budget / 3) ** 0.5)))
    depth = max(1, budget // (3 * width))
    return width, depth


def food_distance_value(gamma: float) -> ValueFn:
    """Learning-free leaf evaluation: the discounted reward for reaching the
    food along an unobstructed shortest path."""
    def value(s: State) -> np.ndarray:
        size = s.size
        dist = np.abs(s.head % size - s.food % size) + np.abs(s.head // size - s.food // size)
        return np.where(s.food >= 0, gamma ** dist, 0.0)
    return value


class Planner:
    def __init__(self, value_fn: ValueFn, budget: int, gamma: float = 0.97):
        self.value_fn, self.budget, self.gamma = value_fn, budget, gamma
        self.width, self.depth = beam_shape(budget)
        self.nodes_used = 0          # simulated moves in the last call (all games)

    def scores(self, root: State) -> np.ndarray:
        """float [G, 3]: the best frontier score reachable after each first action."""
        g_n, k, gamma = len(root), self.width, self.gamma
        beam = root.repeat(k)
        valid = np.zeros(g_n * k, dtype=bool)
        valid[::k] = True
        ret = np.zeros(g_n * k)
        first = np.zeros(g_n * k, dtype=np.int64)
        best = np.full((g_n, 3), -np.inf)       # best finished (dead / won) line per first action
        game = np.repeat(np.arange(g_n), 3 * k)
        self.nodes_used = 0

        for d in range(self.depth):
            kids = beam.repeat(3)
            kvalid = np.repeat(valid, 3)
            kids.alive &= kvalid
            action = np.tile(np.arange(3), g_n * k)
            ate, died = step(kids, action)
            self.nodes_used += int(kvalid.sum())
            kret = np.repeat(ret, 3) + gamma**d * (ate.astype(float) - died.astype(float))
            kfirst = action if d == 0 else np.repeat(first, 3)

            finished = kvalid & ~kids.alive
            np.maximum.at(best, (game[finished], kfirst[finished]), kret[finished])

            live = np.nonzero(kids.alive)[0]
            score = np.full(g_n * 3 * k, -np.inf)
            if live.size:
                score[live] = kret[live] + gamma ** (d + 1) * self.value_fn(kids.take(live))
            if d == self.depth - 1 or not live.size:
                np.maximum.at(best, (game[live], kfirst[live]), score[live])
                break

            per_game = score.reshape(g_n, 3 * k)
            top = np.argsort(-per_game, axis=1, kind="stable")[:, :k]
            flat = (top + np.arange(g_n)[:, None] * 3 * k).ravel()
            beam = kids.take(flat)
            valid = score[flat] > -np.inf
            # Live positions that fall out of the beam remain leaves of the tree.
            chosen = np.zeros(score.shape[0], dtype=bool)
            chosen[flat[valid]] = True
            pruned = live[~chosen[live]]
            np.maximum.at(best, (game[pruned], kfirst[pruned]), score[pruned])
            ret, first = kret[flat], kfirst[flat]
        return best

    def __call__(self, root: State) -> np.ndarray:
        return self.scores(root).argmax(axis=1)
