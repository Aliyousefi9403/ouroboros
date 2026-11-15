import numpy as np

from ouroboros import nn
from ouroboros.engine import from_json, reset, step, to_json
from ouroboros.features import danger, obs11, planes, rotation_table
from ouroboros.verify import analyse


def random_rollout(n=128, steps=300, seed=0):
    rng = np.random.default_rng(seed)
    s = reset(np.arange(n))
    for _ in range(steps):
        yield s
        step(s, rng.integers(0, 3, n))
        dead = np.nonzero(~s.alive)[0]
        if dead.size:
            s.assign(dead, reset(rng.integers(0, 2**31, dead.size)))


def test_engine_invariants():
    for s in random_rollout():
        rows = np.arange(len(s))
        assert ((s.ttl > 0).sum(axis=1) == s.length).all()          # body size
        assert (s.ttl[rows, s.head] == s.length).all()               # head is the newest cell
        assert (np.sort(s.ttl, axis=1)[:, -3:] == np.stack([s.length - 2, s.length - 1, s.length], 1)).all()
        assert (s.ttl[rows, s.food] == 0).all()                      # food on a free cell


def test_determinism_and_tail_rule():
    a = reset([5, 5]); b = reset([5, 5])
    rng = np.random.default_rng(1)
    for _ in range(200):
        act = rng.integers(0, 3, 2)
        act[1] = act[0]
        step(a, act); step(b, act)
    assert (a.ttl == b.ttl).all() and a.rng[0] == a.rng[1] == b.rng[0]
    # A length-4 snake chasing its own tail in a 2x2 loop never dies.
    s = from_json({"size": 10, "ttl": [0] * 100, "head": 0, "dir": 0, "length": 4, "food": 99, "rng": 1})
    s.ttl[0, [0, 10, 11, 1]] = [4, 3, 2, 1]
    for _ in range(12):
        _, died = step(s, [2])
        assert not died[0]


def test_json_round_trip():
    s = next(iter(random_rollout(4, 50)))
    t = from_json(to_json(s, 2))
    assert (t.ttl[0] == s.ttl[2]).all() and t.head[0] == s.head[2] and t.rng[0] == s.rng[2]


def test_danger_predicts_collisions():
    rng = np.random.default_rng(3)
    for s in random_rollout(64, 200, seed=2):
        a = rng.integers(0, 3, len(s))
        nxt = s.copy()
        _, died = step(nxt, a)
        assert (died == danger(s)[np.arange(len(s)), a]).all()


def test_planes_are_egocentric():
    for s in random_rollout(32, 60, seed=4):
        x = planes(s)
        p = s.size + 2
        r, c = np.nonzero(x[0, 3])
        # After rotation the neck (second-newest cell) is always directly below the head.
        ttl_plane = x[:, 2]
        hr, hc = np.unravel_index(x[:, 3].reshape(len(s), -1).argmax(1), (p, p))
        expected = (s.length - 1) / (s.size**2)
        assert np.allclose(ttl_plane[np.arange(len(s)), hr + 1, hc], expected)
    assert sorted(rotation_table(10)[1].tolist()) == list(range(144))


def test_backprop_matches_finite_differences():
    rng = np.random.default_rng(0)
    net = nn.MLP([11, 16, 12, 3], seed=1, dtype=np.float64)
    x = rng.normal(size=(7, 11)); a = rng.integers(0, 3, 7); y = rng.normal(size=7) * 3
    _, grads = nn.q_loss(net, x, a, y)
    eps = 1e-6
    for p, g in zip(net.params, grads):
        for idx in [tuple(rng.integers(0, d) for d in p.shape) for _ in range(12)]:
            old = p[idx]
            p[idx] = old + eps; up, _ = nn.q_loss(net, x, a, y)
            p[idx] = old - eps; down, _ = nn.q_loss(net, x, a, y)
            p[idx] = old
            assert abs((up - down) / (2 * eps) - g[idx]) < 1e-6


def test_adam_fits_a_function():
    rng = np.random.default_rng(0)
    net = nn.MLP([2, 32, 1], seed=0, dtype=np.float64)
    opt = nn.Adam(net.params, lr=1e-2)
    x = rng.uniform(-1, 1, size=(256, 2)); y = np.sin(3 * x[:, 0]) * x[:, 1]
    for _ in range(1500):
        loss, grads = nn.q_loss(net, x, np.zeros(256, dtype=int), y)
        opt.step(net.params, grads)
    assert loss < 2e-3


def test_proofs_on_a_hand_built_trap():
    # Snake coiled so that turning right enters a sealed 1-cell pocket.
    #   . . . .        head at (1,1) heading up; right is (1,2)... build by ttl directly.
    s = from_json({"size": 10, "ttl": [0] * 100, "head": 11, "dir": 0, "length": 9, "food": 99, "rng": 1})
    ring = [11, 21, 22, 23, 13, 3, 2, 1, 0]          # head first; cell 12 is enclosed
    s.ttl[0, ring] = np.arange(9, 0, -1)
    res = analyse(s, limit=10, cap=100000)
    assert res["status"][0, 2] == 2 and res["moves"][0, 2] == 1     # right: one safe move, then dead
    assert res["status"][0, 1] == 2 and res["moves"][0, 1] == 0     # straight: into its own body
    assert res["status"][0, 0] == 1                                  # left: free
