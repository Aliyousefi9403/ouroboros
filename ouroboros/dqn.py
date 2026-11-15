"""Double DQN over the batched engine.

The loop is model-agnostic: a *learner* supplies ``q``, ``target_q``,
``update`` and ``sync``; an *encoder* turns engine states into network input.
The replay buffer stores compact engine states rather than observations, so the
same buffer serves the 11-input MLP and the full-board CNN.
"""
from __future__ import annotations

import time
from typing import Callable, Protocol

import numpy as np

from .engine import State, reset, step
from .play import TRAIN_SEED_MAX, VAL_SEED_BASE, play, stall_limit_for


class Learner(Protocol):
    def q(self, x: np.ndarray) -> np.ndarray: ...
    def target_q(self, x: np.ndarray) -> np.ndarray: ...
    def update(self, x: np.ndarray, actions: np.ndarray, targets: np.ndarray, lr: float) -> float: ...
    def sync(self) -> None: ...
    def save(self, path: str) -> None: ...


class Replay:
    def __init__(self, capacity: int, size: int):
        self.capacity, self.n, self.ptr = capacity, 0, 0
        self.s = reset(np.zeros(capacity), size)
        self.s2 = reset(np.zeros(capacity), size)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

    def add(self, s: State, a, r, s2: State, done) -> None:
        idx = (self.ptr + np.arange(len(s))) % self.capacity
        self.s.assign(idx, s)
        self.s2.assign(idx, s2)
        self.a[idx], self.r[idx], self.done[idx] = a, r, done
        self.ptr = int((self.ptr + len(s)) % self.capacity)
        self.n = min(self.capacity, self.n + len(s))

    def sample(self, batch: int, rng: np.random.Generator):
        idx = rng.integers(0, self.n, size=batch)
        return self.s.take(idx), self.a[idx], self.r[idx], self.s2.take(idx), self.done[idx]


def greedy(learner: Learner, encode: Callable[[State], np.ndarray]):
    return lambda s: learner.q(encode(s)).argmax(axis=1)


def train(
    learner: Learner,
    encode: Callable[[State], np.ndarray],
    out_path: str,
    *,
    size: int = 10,
    n_envs: int = 128,
    total_steps: int = 2_000_000,
    buffer_size: int = 400_000,
    batch: int = 256,
    updates_per_iter: int = 2,
    gamma: float = 0.97,
    lr: float = 5e-4,
    lr_final: float = 5e-5,
    eps_start: float = 1.0,
    eps_final: float = 0.01,
    eps_fraction: float = 0.35,
    warmup: int = 20_000,
    target_sync: int = 1500,
    eval_every: int = 100_000,
    eval_games: int = 100,
    seed: int = 0,
    log: Callable[[dict], None] = print,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    stall = stall_limit_for(size)
    env = reset(rng.integers(0, TRAIN_SEED_MAX, size=n_envs), size)
    buf = Replay(buffer_size, size)
    val_seeds = VAL_SEED_BASE + np.arange(eval_games)
    history: list[dict] = []
    best, updates, next_eval, losses, t0 = -1.0, 0, eval_every, [], time.time()

    for it in range(total_steps // n_envs):
        steps_done = (it + 1) * n_envs
        frac = steps_done / total_steps
        eps = max(eps_final, eps_start + (eps_final - eps_start) * frac / eps_fraction)
        cur_lr = lr + (lr_final - lr) * frac

        actions = learner.q(encode(env)).argmax(axis=1)
        explore = rng.random(n_envs) < eps
        actions[explore] = rng.integers(0, 3, size=int(explore.sum()))
        before = env.copy()
        ate, died = step(env, actions, stall)
        reward = ate.astype(np.float32) - died.astype(np.float32)
        done = ~env.alive
        buf.add(before, actions, reward, env, done.astype(np.float32))
        if done.any():
            idx = np.nonzero(done)[0]
            env.assign(idx, reset(rng.integers(0, TRAIN_SEED_MAX, size=len(idx)), size))

        if buf.n >= warmup:
            for _ in range(updates_per_iter):
                s, a, r, s2, d = buf.sample(batch, rng)
                x2 = encode(s2)
                a2 = learner.q(x2).argmax(axis=1)                       # online net picks...
                q2 = learner.target_q(x2)[np.arange(batch), a2]         # ...target net scores
                targets = r + gamma * (1.0 - d) * q2
                losses.append(learner.update(encode(s), a, targets.astype(np.float32), cur_lr))
                updates += 1
                if updates % target_sync == 0:
                    learner.sync()

        if steps_done >= next_eval or it == total_steps // n_envs - 1:
            next_eval += eval_every
            res = play(greedy(learner, encode), val_seeds, size)
            mean = float(res["score"].mean())
            rec = {
                "steps": steps_done, "eps": round(eps, 3), "val_mean_score": round(mean, 2),
                "val_max_score": int(res["score"].max()),
                "loss": round(float(np.mean(losses[-500:])), 5) if losses else None,
                "minutes": round((time.time() - t0) / 60, 1),
            }
            history.append(rec)
            log(rec)
            if mean > best:
                best = mean
                learner.save(out_path)
    return history
