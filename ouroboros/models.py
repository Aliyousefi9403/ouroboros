"""The two learned agents and their on-disk formats.

ReflexLearner  - 11 inputs -> MLP, trained with the from-scratch NumPy library.
BoardLearner   - full egocentric board -> small CNN, trained with PyTorch.

``board_forward_numpy`` is a dependency-free reference implementation of the
CNN's forward pass. It documents the exact arithmetic the browser performs and
is tested against PyTorch.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import nn as scratch
from .features import N_PLANES, obs11, planes

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


# --------------------------------------------------------------------------
# 11-input reflex agent (NumPy from scratch)
# --------------------------------------------------------------------------
class ReflexLearner:
    encode = staticmethod(obs11)

    def __init__(self, hidden=(128, 128), seed: int = 0):
        self.net = scratch.MLP([11, *hidden, 3], seed=seed)
        self.target = scratch.MLP([11, *hidden, 3], seed=seed)
        self.target.copy_from(self.net)
        self.opt = scratch.Adam(self.net.params)

    def q(self, x):
        return self.net.forward(x)

    def target_q(self, x):
        return self.target.forward(x)

    def update(self, x, actions, targets, lr):
        self.opt.lr = lr
        loss, grads = scratch.q_loss(self.net, x, actions, targets)
        self.opt.step(self.net.params, grads)
        return float(loss)

    def sync(self):
        self.target.copy_from(self.net)

    def save(self, path):
        np.savez(path, **self.net.state_dict())

    @classmethod
    def load(cls, path=None) -> "ReflexLearner":
        self = cls.__new__(cls)
        with np.load(path or os.path.join(MODEL_DIR, "reflex.npz")) as d:
            self.net = scratch.MLP.from_state_dict({k: d[k] for k in d.files})
        return self


# --------------------------------------------------------------------------
# Full-board agent (PyTorch for training)
# --------------------------------------------------------------------------
def _torch():
    import torch
    return torch


def make_board_net(size: int = 10):
    torch = _torch()
    final = size + 2 - 4  # one padded conv keeps P, two valid convs shrink by 2 each
    return torch.nn.Sequential(
        torch.nn.Conv2d(N_PLANES, 16, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, 3), torch.nn.ReLU(),
        torch.nn.Conv2d(32, 32, 3), torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * final * final, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 3),
    )


class BoardLearner:
    encode = staticmethod(planes)

    def __init__(self, size: int = 10, seed: int = 0):
        torch = _torch()
        torch.manual_seed(seed)
        self.size = size
        self.net = make_board_net(size)
        self.target = make_board_net(size)
        self.target.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=5e-4)

    def q(self, x):
        torch = _torch()
        with torch.no_grad():
            return self.net(torch.from_numpy(x)).numpy()

    def target_q(self, x):
        torch = _torch()
        with torch.no_grad():
            return self.target(torch.from_numpy(x)).numpy()

    def update(self, x, actions, targets, lr):
        torch = _torch()
        for g in self.opt.param_groups:
            g["lr"] = lr
        q = self.net(torch.from_numpy(x)).gather(1, torch.from_numpy(actions)[:, None])[:, 0]
        loss = torch.nn.functional.smooth_l1_loss(q, torch.from_numpy(targets))
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        return loss.item()

    def sync(self):
        self.target.load_state_dict(self.net.state_dict())

    def save(self, path):
        np.savez(path, **{k.replace(".", "_"): v.numpy() for k, v in self.net.state_dict().items()})

    @classmethod
    def load(cls, path=None, size: int = 10) -> "BoardLearner":
        torch = _torch()
        self = cls.__new__(cls)
        self.size = size
        self.net = make_board_net(size)
        with np.load(path or os.path.join(MODEL_DIR, "board.npz")) as d:
            self.net.load_state_dict({k.replace("_", ".", 1): torch.from_numpy(d[k]) for k in d.files})
        self.net.eval()
        return self

    def value(self, x):
        """State value V(s) = max_a Q(s, a), used by the planner at the frontier."""
        return self.q(x).max(axis=1)


def _conv3x3(x: np.ndarray, w: np.ndarray, b: np.ndarray, pad: int) -> np.ndarray:
    """x [B, C, H, W], w [O, C, 3, 3] -> [B, O, H', W'] (cross-correlation, as in PyTorch)."""
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    win = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(2, 3))  # [B, C, H', W', 3, 3]
    return np.einsum("bchwij,ocij->bohw", win, w, optimize=True) + b[None, :, None, None]


def board_forward_numpy(weights: dict, x: np.ndarray) -> np.ndarray:
    """Reference forward pass of the board network from a saved ``board.npz`` dict."""
    h = np.maximum(_conv3x3(x, weights["0_weight"], weights["0_bias"], 1), 0)
    h = np.maximum(_conv3x3(h, weights["2_weight"], weights["2_bias"], 0), 0)
    h = np.maximum(_conv3x3(h, weights["4_weight"], weights["4_bias"], 0), 0)
    h = h.reshape(h.shape[0], -1)
    h = np.maximum(h @ weights["7_weight"].T + weights["7_bias"], 0)
    return h @ weights["9_weight"].T + weights["9_bias"]


# --------------------------------------------------------------------------
# Export for the browser: one little-endian float32 blob + a JSON manifest.
# --------------------------------------------------------------------------
def export_web(npz_path: str, out_dir: str, name: str, meta: dict) -> None:
    with np.load(npz_path) as d:
        tensors = [(k, d[k].astype("<f4")) for k in d.files]
    manifest, offset = {"name": name, **meta, "tensors": []}, 0
    with open(os.path.join(out_dir, f"{name}.bin"), "wb") as f:
        for k, t in tensors:
            f.write(t.tobytes())
            manifest["tensors"].append({"name": k, "shape": list(t.shape), "offset": offset})
            offset += t.size
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(manifest, f, indent=1)
