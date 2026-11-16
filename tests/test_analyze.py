"""Tests for the tiering rule and stability metrics in phase0/analyze.py."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase0"))
import analyze  # noqa: E402
from analyze import COLD, HOT, WARM, plans, tier_rule  # noqa: E402


def naive_rule(m, q, eps_w, eps_c):
    """Rule B written out literally, one head at a time, as a reference."""
    nb = len(m)
    hot = {0, max(nb - 2, 0), nb - 1}
    rest = sorted((b for b in range(nb) if b not in hot), key=lambda b: -q[b])
    for b in rest:
        if sum(q[x] for x in range(nb) if x not in hot) <= eps_w:
            break
        hot.add(b)
    cold, cum = set(), 0.0
    for b in sorted((b for b in range(nb) if b not in hot), key=lambda b: m[b]):
        if cum + m[b] >= eps_c:
            break
        cum += m[b]
        cold.add(b)
    return [HOT if b in hot else COLD if b in cold else WARM for b in range(nb)]


def block_stats(w, block=16):
    n = len(w)
    nb = -(-n // block)
    wp = np.zeros(nb * block)
    wp[:n] = w
    wb = wp.reshape(nb, block)
    return torch.tensor(wb.sum(1)), torch.tensor((wb ** 2).sum(1))


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("eps_w,eps_c", [(1e-2, 0.005), (1e-3, 0.02), (1e-4, 0.05)])
def test_matches_naive(seed, eps_w, eps_c):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(40, 2000))
    logits = rng.standard_normal(n) * rng.choice([0.5, 3, 8])
    w = np.exp(logits - logits.max())
    w /= w.sum()
    m, q = block_stats(w)
    got = tier_rule(m, q, eps_w, eps_c).tolist()
    assert got == naive_rule(m.tolist(), q.tolist(), eps_w, eps_c)


def test_diffuse_head_is_warm():
    n = 1024
    m, q = block_stats(np.full(n, 1 / n))  # 64 blocks, m = 1/64 each, q tiny
    t = tier_rule(m, q, 1e-3, 0.05)
    assert (t == HOT).sum() == 3 and t[0] == HOT and t[-1] == HOT and t[-2] == HOT
    assert (t == COLD).sum() == 3  # 3/64 = 0.047 < 0.05, 4/64 is not
    assert (t == WARM).sum() == 58


def test_sharp_head_promotes_spike():
    n = 1024
    w = np.full(n, 0.1 / (n - 1))
    w[170] = 0.9  # block 10
    m, q = block_stats(w)
    t = tier_rule(m, q, 1e-3, 0.05)
    assert t[10] == HOT and (t == HOT).sum() == 4


def test_gqa_exact_takes_max_tier_and_differs_from_max_reduced():
    # Two query heads sharing a KV head: each wants a different block HOT.
    n, nb = 512, 32
    w1 = np.full(n, 0.2 / n); w1[16 * 5] += 0.8
    w2 = np.full(n, 0.2 / n); w2[16 * 20] += 0.8
    s = [block_stats(w) for w in (w1, w2)]
    m = torch.stack([x[0] for x in s])[None, None]  # [L=1, H=1, G=2, nb]
    q = torch.stack([x[1] for x in s])[None, None]
    exact = plans(m, q, "exact", [(1e-3, 0.05)])[0, 0, 0]
    assert exact[5] == HOT and exact[20] == HOT
    per_head = [tier_rule(m[0, 0, g], q[0, 0, g], 1e-3, 0.05) for g in range(2)]
    assert torch.equal(exact, torch.maximum(*per_head))
    assert plans(m, q, "max_reduced", [(1e-3, 0.05)]).shape == (1, 1, 1, nb)


def test_lag_stats_identical_plans():
    rng = np.random.default_rng(0)
    L, H, G, nb = 2, 2, 3, 40
    m = torch.tensor(rng.dirichlet(np.ones(nb) * 0.3, size=(L, H, G)))
    q = m ** 2 / 4
    cfgs = [(1e-3, 0.05), (1e-2, 0.02)]
    t = plans(m, q, "exact", cfgs)
    acc = analyze.Acc(len(cfgs), L, H)
    analyze.lag_stats(acc, 1, t, t, m, torch.tensor([c for _, c in cfgs], dtype=torch.float64))
    assert np.allclose(acc.jac_sum[:, 0] / acc.n_kv_lag[0], 1.0)
    assert acc.missed_sum[:, 0].sum() == 0
    # per query head cold mass under the KV-head plan stays below eps_c
    assert (acc.sc_max[:, 0] < np.array([c for _, c in cfgs])).all()


def test_stale_plan_padded_with_hot():
    L, H, G = 1, 1, 1
    m_old = torch.full((L, H, G, 10), 0.1)
    m_new = torch.zeros((L, H, G, 12)); m_new[..., 11] = 1.0
    cfg = [(1e-3, 0.5)]
    stale = plans(m_old, m_old ** 2 / 16, "exact", cfg)
    fresh = plans(m_new, m_new ** 2 / 16, "exact", cfg)
    acc = analyze.Acc(1, L, H)
    analyze.lag_stats(acc, 2, stale, fresh, m_new, torch.tensor([0.5], dtype=torch.float64))
    assert acc.sc_sum[0, 1] == 0.0  # mass in new tail blocks never counts as COLD
    assert acc.missed_sum[0, 1] == 0


def test_rho_formula():
    fr = np.zeros(3); fr[HOT], fr[WARM], fr[COLD] = 0.1, 0.5, 0.4
    assert analyze.rho(fr, 4, 4) == pytest.approx(0.1 + 0.5 * 0.25 + 0.4 * 0.5 * 0.25 / 4)
