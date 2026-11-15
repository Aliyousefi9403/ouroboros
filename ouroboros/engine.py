"""Deterministic, batched Snake engine.

The whole game state is a handful of NumPy arrays with a leading batch axis, so
the same code drives a single interactive game, 256 parallel training games and
the thousands of hypothetical futures a planner simulates.

Representation
--------------
The board is stored as a *time-to-live* grid: ``ttl[cell]`` is the number of
moves until that cell is vacated (0 = empty, 1 = tail tip, ``length`` = head).
This single array encodes the body in head-to-tail order, makes a move an O(1)
vectorised operation (decrement everything, write the new head) and gives the
exact collision rule for free: a cell is deadly iff its ttl is still positive
*after* the tail has moved.

Determinism
-----------
Food is drawn from a 32-bit mulberry32 stream that is part of the state. The
JavaScript engine in ``docs/js/engine.js`` implements the identical arithmetic,
so a (seed, action list) pair replays bit-for-bit in the browser.

Directions are absolute (0=up, 1=right, 2=down, 3=left); actions are relative
(0=turn left, 1=straight, 2=turn right), so every action is always legal.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

DX = np.array([0, 1, 0, -1], dtype=np.int32)
DY = np.array([-1, 0, 1, 0], dtype=np.int32)

LEFT, STRAIGHT, RIGHT = 0, 1, 2
ACTION_NAMES = ("left", "straight", "right")


def rng_next(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One mulberry32 step on a uint32 array. Returns (new_state, output)."""
    with np.errstate(over="ignore"):
        s = (state + np.uint32(0x6D2B79F5)).astype(np.uint32)
        t = ((s ^ (s >> np.uint32(15))) * (s | np.uint32(1))).astype(np.uint32)
        t = ((t + ((t ^ (t >> np.uint32(7))) * (t | np.uint32(61))).astype(np.uint32)) ^ t).astype(np.uint32)
        out = (t ^ (t >> np.uint32(14))).astype(np.uint32)
    return s, out


@dataclass
class State:
    """A batch of N independent games on a ``size`` x ``size`` board."""

    size: int
    ttl: np.ndarray      # int16 [N, size*size]  moves until each cell clears
    head: np.ndarray     # int32 [N]             flat index of the head
    dir: np.ndarray      # int8  [N]             absolute heading
    length: np.ndarray   # int16 [N]
    food: np.ndarray     # int32 [N]             flat index, -1 when board is full
    rng: np.ndarray      # uint32 [N]            food stream state
    alive: np.ndarray    # bool  [N]
    won: np.ndarray      # bool  [N]             filled the board
    since: np.ndarray    # int32 [N]             moves since last food

    _ARRAYS = ("ttl", "head", "dir", "length", "food", "rng", "alive", "won", "since")

    def __len__(self) -> int:
        return self.head.shape[0]

    @property
    def score(self) -> np.ndarray:
        return self.length.astype(np.int32) - 3

    def copy(self) -> "State":
        return State(self.size, **{k: getattr(self, k).copy() for k in self._ARRAYS})

    def take(self, idx) -> "State":
        """Fancy-index the batch axis (always returns a copy)."""
        return State(self.size, **{k: getattr(self, k)[idx].copy() for k in self._ARRAYS})

    def repeat(self, k: int) -> "State":
        """Each game repeated k times consecutively: [g0, g0, g0, g1, ...]."""
        return State(self.size, **{k_: np.repeat(getattr(self, k_), k, axis=0) for k_ in self._ARRAYS})

    def assign(self, idx, other: "State") -> None:
        for k in self._ARRAYS:
            getattr(self, k)[idx] = getattr(other, k)


def _place_food(s: State, rows: np.ndarray) -> None:
    """Draw food for the given rows: the (r mod n)-th free cell in row-major order."""
    if rows.size == 0:
        return
    s.rng[rows], r = rng_next(s.rng[rows])
    free = s.ttl[rows] == 0
    n = free.sum(axis=1)
    full = n == 0
    k = (r % np.maximum(n, 1).astype(np.uint32)).astype(np.int64)
    cell = (np.cumsum(free, axis=1) > k[:, None]).argmax(axis=1)
    s.food[rows] = np.where(full, -1, cell)
    s.won[rows[full]] = True
    s.alive[rows[full]] = False


def reset(seeds, size: int = 10) -> State:
    """Start one game per seed: a length-3 snake in the middle, heading right."""
    seeds = np.atleast_1d(np.asarray(seeds)).astype(np.uint32)
    n = seeds.shape[0]
    ttl = np.zeros((n, size * size), dtype=np.int16)
    row, col = size // 2, size // 2
    head = row * size + col
    ttl[:, head] = 3
    ttl[:, head - 1] = 2
    ttl[:, head - 2] = 1
    s = State(
        size=size,
        ttl=ttl,
        head=np.full(n, head, dtype=np.int32),
        dir=np.ones(n, dtype=np.int8),
        length=np.full(n, 3, dtype=np.int16),
        food=np.full(n, -1, dtype=np.int32),
        rng=seeds.copy(),
        alive=np.ones(n, dtype=bool),
        won=np.zeros(n, dtype=bool),
        since=np.zeros(n, dtype=np.int32),
    )
    _place_food(s, np.arange(n))
    return s


def step(s: State, actions, stall_limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Advance every live game by one move, in place.

    Returns ``(ate, died)`` boolean arrays. Finished games are left untouched.
    ``stall_limit`` ends a game (as a death) after that many foodless moves;
    it is a training/evaluation convenience and is off for interactive play.
    """
    size = s.size
    actions = np.asarray(actions)
    act = s.alive.copy()
    d = np.where(act, (s.dir + actions - 1) & 3, s.dir).astype(np.int8)
    nx = s.head % size + DX[d]
    ny = s.head // size + DY[d]
    inb = (nx >= 0) & (nx < size) & (ny >= 0) & (ny < size)
    new = np.where(inb, ny * size + nx, 0)
    ate = act & inb & (new == s.food)

    # The tail moves first (unless we are growing), so its cell is enterable.
    move = act & inb & ~ate
    sub = s.ttl[move]
    sub -= sub > 0
    rows = np.arange(len(s))
    hit = np.zeros(len(s), dtype=bool)
    hit[move] = sub[np.arange(sub.shape[0]), new[move]] > 0
    died = act & (~inb | hit)
    ok = act & ~died
    okm = ok[move]
    s.ttl[rows[move][okm]] = sub[okm]          # commit tail movement only for survivors

    s.length[ate] += 1
    s.ttl[rows[ok], new[ok]] = s.length[ok]
    s.head[ok] = new[ok]
    s.dir[ok] = d[ok]
    s.alive[died] = False
    s.since[ok] += 1
    s.since[ate] = 0
    _place_food(s, rows[ate])

    if stall_limit is not None:
        stalled = s.alive & (s.since >= stall_limit)
        s.alive[stalled] = False
        died = died | stalled
    return ate, died


def body_cells(s: State, i: int = 0) -> list[int]:
    """Flat cell indices of game ``i``'s snake, head first."""
    ttl = s.ttl[i]
    cells = np.nonzero(ttl)[0]
    return cells[np.argsort(-ttl[cells])].tolist()


def render(s: State, i: int = 0) -> str:
    size = s.size
    ttl = s.ttl[i]
    out = []
    for r in range(size):
        line = ""
        for c in range(size):
            j = r * size + c
            if j == s.head[i] and ttl[j] > 0:
                line += "@"
            elif ttl[j] > 0:
                line += "o"
            elif j == s.food[i]:
                line += "*"
            else:
                line += "."
        out.append(line)
    return "\n".join(out)


def concat(states: list[State]) -> State:
    return State(states[0].size, **{k: np.concatenate([getattr(s, k) for s in states]) for k in State._ARRAYS})


def to_json(s: State, i: int = 0) -> dict:
    """One game as plain JSON (the format docs/js/engine.js ``fromJSON`` reads)."""
    return {
        "size": s.size, "ttl": s.ttl[i].tolist(), "head": int(s.head[i]), "dir": int(s.dir[i]),
        "length": int(s.length[i]), "food": int(s.food[i]), "rng": int(s.rng[i]),
    }


def from_json(o: dict) -> State:
    s = reset([0], o["size"])
    s.ttl[0] = o["ttl"]
    s.head[0], s.dir[0], s.length[0], s.food[0], s.rng[0] = o["head"], o["dir"], o["length"], o["food"], o["rng"]
    return s
