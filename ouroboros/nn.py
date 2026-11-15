"""A small neural-network library written from scratch in NumPy.

Everything needed to train the 11-input Q-network lives here: the forward
pass, hand-derived backpropagation, the Huber TD loss and Adam. The gradients
are verified against central finite differences in ``tests/test_nn.py``.
"""
from __future__ import annotations

import numpy as np


class MLP:
    """Fully connected network with ReLU hidden layers and a linear output."""

    def __init__(self, sizes, seed: int = 0, dtype=np.float32):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.params: list[np.ndarray] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            w = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=(fan_in, fan_out))  # He init
            self.params += [w.astype(dtype), np.zeros(fan_out, dtype=dtype)]

    # -- inference ---------------------------------------------------------
    def forward(self, x: np.ndarray, cache: list | None = None) -> np.ndarray:
        """x: [B, in] -> [B, out]. If ``cache`` is a list, activations are
        appended to it for a subsequent ``backward`` call."""
        n_layers = len(self.params) // 2
        h = x
        for i in range(n_layers):
            w, b = self.params[2 * i], self.params[2 * i + 1]
            if cache is not None:
                cache.append(h)
            h = h @ w + b
            if i < n_layers - 1:
                h = np.maximum(h, 0.0)
        return h

    # -- learning ----------------------------------------------------------
    def backward(self, cache: list, dout: np.ndarray) -> list[np.ndarray]:
        """Backpropagate dLoss/dOutput. Returns gradients aligned with ``params``.

        For layer i with input a_i and pre-activation z_i = a_i W_i + b_i:
            dW_i = a_i^T dz_i,  db_i = sum(dz_i),  da_i = dz_i W_i^T,
            dz_{i-1} = da_i * [a_i > 0]          (a_i is a ReLU output)
        """
        n_layers = len(self.params) // 2
        grads: list[np.ndarray] = [None] * len(self.params)  # type: ignore[list-item]
        dz = dout
        for i in reversed(range(n_layers)):
            a = cache[i]
            grads[2 * i] = a.T @ dz
            grads[2 * i + 1] = dz.sum(axis=0)
            if i > 0:
                dz = (dz @ self.params[2 * i].T) * (a > 0)
        return grads

    def copy_from(self, other: "MLP") -> None:
        self.params = [p.copy() for p in other.params]

    def state_dict(self) -> dict:
        return {f"p{i}": p for i, p in enumerate(self.params)}

    @classmethod
    def from_state_dict(cls, d: dict) -> "MLP":
        params = [d[f"p{i}"] for i in range(len(d))]
        net = cls.__new__(cls)
        net.params = [np.asarray(p) for p in params]
        net.sizes = [params[0].shape[0]] + [params[i].shape[0] for i in range(1, len(params), 2)]
        return net


def huber(pred: np.ndarray, target: np.ndarray, delta: float = 1.0):
    """Mean Huber loss and its gradient with respect to ``pred``."""
    err = pred - target
    small = np.abs(err) <= delta
    loss = np.where(small, 0.5 * err**2, delta * (np.abs(err) - 0.5 * delta))
    grad = np.where(small, err, delta * np.sign(err)) / err.shape[0]
    return loss.mean(), grad


def q_loss(net: MLP, obs: np.ndarray, actions: np.ndarray, targets: np.ndarray):
    """TD loss on the Q-value of the taken action. Returns (loss, grads)."""
    cache: list = []
    q = net.forward(obs, cache)
    rows = np.arange(obs.shape[0])
    loss, g = huber(q[rows, actions], targets)
    dout = np.zeros_like(q)
    dout[rows, actions] = g
    return loss, net.backward(cache, dout)


class Adam:
    def __init__(self, params, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads) -> None:
        self.t += 1
        corr = self.lr * np.sqrt(1 - self.b2**self.t) / (1 - self.b1**self.t)
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= corr * m / (np.sqrt(v) + self.eps)
