"""Two views of the same game state.

``obs11``   - the classic 11-number observation (3 danger bits, 4 heading bits,
              4 food-direction bits). It is what most tutorial Snake agents see.
``planes``  - the full board, egocentric (rotated so the snake always heads
              "up") with the body encoded in head-to-tail order as
              time-to-clear. This is what the convolutional agent sees.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .engine import DX, DY, State

N_PLANES = 5  # wall, occupied, time-to-clear, head, food


def danger(s: State) -> np.ndarray:
    """bool [N, 3]: would relative action (left, straight, right) collide next move?"""
    size = s.size
    rows = np.arange(len(s))
    out = np.zeros((len(s), 3), dtype=bool)
    for a in range(3):
        d = (s.dir + a - 1) & 3
        nx = s.head % size + DX[d]
        ny = s.head // size + DY[d]
        inb = (nx >= 0) & (nx < size) & (ny >= 0) & (ny < size)
        cell = np.where(inb, ny * size + nx, 0)
        # ttl == 1 is the tail tip, which will have moved away.
        out[:, a] = ~inb | (s.ttl[rows, cell] > 1)
    return out


def obs11(s: State) -> np.ndarray:
    """float32 [N, 11] in the original ordering:
    danger(straight, right, left), heading(left, right, up, down),
    food(left, right, up, down)."""
    size = s.size
    dg = danger(s)
    hx, hy = s.head % size, s.head // size
    fx, fy = s.food % size, s.food // size
    has = s.food >= 0
    o = np.stack(
        [
            dg[:, 1], dg[:, 2], dg[:, 0],
            s.dir == 3, s.dir == 1, s.dir == 0, s.dir == 2,
            has & (fx < hx), has & (fx > hx), has & (fy < hy), has & (fy > hy),
        ],
        axis=1,
    )
    return o.astype(np.float32)


def obs_key(o: np.ndarray) -> np.ndarray:
    """Pack each 11-bit observation into an integer (bit i = feature i)."""
    return (o.astype(np.int64) << np.arange(o.shape[1])).sum(axis=1)


@lru_cache(maxsize=None)
def rotation_table(size: int) -> np.ndarray:
    """int [4, P*P] gather indices (P = size + 2, the wall-padded board).

    ``padded.flat[table[d]]`` is the board rotated so heading ``d`` points up.
    """
    p = size + 2
    r, c = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    src = [
        (r, c),                    # heading up: identity
        (c, p - 1 - r),            # heading right: rotate 90 deg counter-clockwise
        (p - 1 - r, p - 1 - c),    # heading down: rotate 180 deg
        (p - 1 - c, r),            # heading left: rotate 90 deg clockwise
    ]
    return np.stack([(sr * p + sc).ravel() for sr, sc in src]).astype(np.int64)


def planes(s: State) -> np.ndarray:
    """float32 [N, 5, P, P] egocentric board tensor (P = size + 2)."""
    size = s.size
    p = size + 2
    n = len(s)
    rows = np.arange(n)
    x = np.zeros((n, N_PLANES, p, p), dtype=np.float32)
    x[:, 0, 0, :] = x[:, 0, -1, :] = x[:, 0, :, 0] = x[:, 0, :, -1] = 1.0
    ttl = s.ttl.reshape(n, size, size)
    x[:, 1, 1:-1, 1:-1] = ttl > 0
    x[:, 2, 1:-1, 1:-1] = ttl / float(size * size)
    x[rows, 3, s.head // size + 1, s.head % size + 1] = 1.0
    has = s.food >= 0
    x[rows[has], 4, s.food[has] // size + 1, s.food[has] % size + 1] = 1.0
    idx = rotation_table(size)[s.dir]                       # [N, P*P]
    flat = x.reshape(n, N_PLANES, p * p)
    out = np.take_along_axis(flat, np.broadcast_to(idx[:, None, :], flat.shape), axis=2)
    return out.reshape(n, N_PLANES, p, p)
