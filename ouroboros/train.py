"""Train an agent:  python -m ouroboros.train {reflex,board} [--steps N]"""
from __future__ import annotations

import argparse
import json
import os

from .dqn import train
from .models import MODEL_DIR, BoardLearner, ReflexLearner

CONFIGS = {
    "reflex": dict(total_steps=1_500_000, n_envs=128, batch=256, lr=1e-3, lr_final=1e-4, gamma=0.95),
    "board": dict(total_steps=6_000_000, n_envs=128, batch=256, lr=5e-4, lr_final=5e-5, gamma=0.97,
                  eps_fraction=0.25, eval_every=200_000),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", choices=CONFIGS)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = dict(CONFIGS[args.agent], size=args.size, seed=args.seed)
    if args.steps:
        cfg["total_steps"] = args.steps
    learner = ReflexLearner(seed=args.seed) if args.agent == "reflex" else BoardLearner(args.size, seed=args.seed)
    os.makedirs(MODEL_DIR, exist_ok=True)
    out = os.path.join(MODEL_DIR, f"{args.agent}.npz")
    history = train(learner, learner.encode, out, log=lambda r: print(json.dumps(r), flush=True), **cfg)
    with open(os.path.join(MODEL_DIR, f"{args.agent}_training.json"), "w") as f:
        json.dump({"config": cfg, "history": history}, f, indent=1)


if __name__ == "__main__":
    main()
