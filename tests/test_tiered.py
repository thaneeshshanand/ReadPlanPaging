"""Tests for the part D simulator (phase0/tiered.py) and its read accounting (analyze_d.py)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase0"))
import analyze_d  # noqa: E402
import tiered  # noqa: E402
from analyze import COLD, HOT, WARM  # noqa: E402
from tiered import BLOCK, CoarseStore, Variant, dequantise, quantise  # noqa: E402

H, G, D = 2, 3, 32


def make_inputs(n, seed=0, spike=None):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, H * G, 1, D, generator=g)
    k = torch.randn(1, H, n, D, generator=g)
    v = torch.randn(1, H, n, D, generator=g)
    if spike is not None:  # make token `spike` dominate every head's attention
        k[0, :, spike] = q[0, :: G, 0] * 3
    return q.half(), k.half(), v.half()


def run(variant, q, k, v, layers=1, step=0, state=None):
    st = tiered.STATE
    if state is None:
        st.L = layers
        st.reset(variant, torch.device("cpu"))
    st.step = step
    st.begin_step()
    mod = SimpleNamespace(layer_idx=0)
    return tiered.tiered_attention(mod, q, k, v, None, None).float()


def exact(q, k, v):
    qv = q[0, :, 0].float().view(H, G, D)
    w = torch.softmax(torch.bmm(qv, k[0].float().transpose(1, 2)) * D**-0.5, -1)
    return torch.bmm(w, v[0].float()).reshape(1, 1, H * G, D)


@pytest.mark.parametrize("bits", [8, 4, 1])
def test_quantiser_error_bounded_by_half_step(bits):
    x = torch.randn(3, 5, BLOCK, D)
    code, scale, lo = quantise(x, bits, dim=2)
    assert code.max() <= 2**bits - 1
    err = (dequantise(code, scale, lo) - x).abs()
    assert (err <= scale.float() / 2 + 1e-2 * scale.float() + 1e-3).all()


def test_coarse_store_incremental_equals_one_shot():
    k = torch.randn(H, 100, D).half()
    a, b = CoarseStore(4, values=True), CoarseStore(4, values=True)
    a.update(k[:, :40], k[:, :40])
    a.update(k, k)
    b.update(k, k)
    assert a.nbf == b.nbf == 6
    assert torch.equal(a.full("k", k), b.full("k", k)) and torch.equal(a.full("v", k), b.full("v", k))
    assert torch.equal(a.full("k", k)[:, 96:], k[:, 96:].float())  # partial block stays fp16


def test_fp16_variant_is_exact():
    q, k, v = make_inputs(200)
    assert torch.allclose(run(Variant("fp16"), q, k, v), exact(q, k, v), atol=2e-3)


def test_all_hot_plan_is_exact():
    q, k, v = make_inputs(200)
    out = run(Variant("fresh", eps_w=-1.0, eps_c=0.0), q, k, v)  # eps_w < 0 promotes every block
    assert torch.allclose(out, exact(q, k, v), atol=2e-3)
    assert tiered.STATE.counts[0, 0] == H * 13


def test_hot_only_drops_non_hot_blocks():
    n, spike = 320, 100
    q, k, v = make_inputs(n, spike=spike)
    out = run(Variant("hot_only", eps_w=1e-3), q, k, v)
    c = tiered.STATE.counts[0]
    assert c[1] == 0 and c[0] + c[2] == c[5]  # no WARM, HOT + COLD = all
    # recompute: attention restricted to HOT tokens
    nb = n // BLOCK
    qv = q[0, :, 0].float().view(H, G, D)
    w = torch.softmax(torch.bmm(qv, k[0].float().transpose(1, 2)) * D**-0.5, -1)
    tiers = tiered.rule_b(w, nb, 1e-3, 0.0)
    assert (tiers[:, spike // BLOCK] == HOT).all()
    s = torch.bmm(qv, k[0].float().transpose(1, 2)) * D**-0.5
    s = s.masked_fill((tiers != HOT).repeat_interleave(BLOCK, 1)[:, None, :], float("-inf"))
    ref = torch.bmm(torch.softmax(s, -1), v[0].float()).reshape(1, 1, H * G, D)
    assert torch.allclose(out, ref, atol=2e-3)


def test_stale_plan_reused_and_tail_blocks_hot():
    q, k, v = make_inputs(200)
    var = Variant("stale", R=4, eps_w=1e-3, eps_c=0.05)
    run(var, q, k[:, :, :160], v[:, :, :160], step=0)
    plan = tiered.STATE.layers[0]["plan"].clone()
    st = tiered.STATE
    st.step = 1
    st.begin_step()
    tiered.tiered_attention(SimpleNamespace(layer_idx=0), q, k, v, None, None)
    assert torch.equal(st.layers[0]["plan"], plan)  # not refreshed at step 1
    assert st.counts[0, 0] >= (plan == HOT).sum() + H * (13 - 10)  # 3 new blocks are HOT


def test_promotion_recovers_spike_hidden_in_cold_block():
    # Keys never change; the query turns toward a token in a block the plan made COLD.
    n, spike = 640, 200
    q, k, v = make_inputs(n)
    blk = spike // BLOCK
    u = torch.randn(H, D, generator=torch.Generator().manual_seed(1))
    for h in range(H):
        Q = q[0, h * G : (h + 1) * G, 0].float()  # [G, D]
        u[h] -= Q.T @ torch.linalg.solve(Q @ Q.T, Q @ u[h])  # orthogonal to every step-0 query
        u[h] = 3 * u[h] / u[h].norm() * D**0.5 / 2
        k[0, h, blk * BLOCK : (blk + 1) * BLOCK] = -Q.sum(0).half()
        k[0, h, spike] = u[h].half()
    var = Variant("promote", R=8, eps_w=1e-3, eps_c=0.5)
    run(var, q, k, v, step=0)
    assert (tiered.STATE.layers[0]["plan"][:, blk] == COLD).all()
    q2 = u.repeat_interleave(G, 0).view(1, H * G, 1, D).half()
    st = tiered.STATE
    st.step = 1
    st.begin_step()
    tiered.tiered_attention(SimpleNamespace(layer_idx=0), q2, k, v, None, None)
    ev_c, rec_c = st.recall[0, 0], st.recall[0, 1]
    assert ev_c >= H and rec_c == ev_c  # every KV head's missed block was promoted
    assert st.counts[0, 4] >= H  # promoted from COLD
    assert st.tail[0, 7] < st.tail[0, 3]  # fewer query heads with > 10% mass in COLD after promotion


def test_twotier_has_no_cold():
    q, k, v = make_inputs(300)
    run(Variant("twotier", R=4, eps_w=1e-3, eps_c=0.0), q, k, v)
    assert tiered.STATE.counts[0, 2] == 0


def test_rho_accounting():
    fr = {"hot": 0.1, "warm": 0.5, "cold": 0.4, "prom_warm": 0.02, "prom_cold": 0.01}
    v = {"kind": "promote", "bits": 4, "R": 4, "scan_bits": 1}
    expect = 0.1 + 0.48 * 0.25 + 0.02 * (0.125 + 1) + 0.4 / 32 + 0.01 + 0.4 * 0.125 / 4
    assert analyze_d.rho(v, fr, False, 128) == pytest.approx(expect)
    stale = {"kind": "stale", "bits": 4, "R": 4}
    assert analyze_d.rho(stale, fr, False, 128) == pytest.approx(0.1 + 0.5 * 0.25 + 0.4 * 0.5 * 0.25 / 4)  # PHASE0 E
    uni = {"kind": "uniform", "bits": 4, "R": 1}
    allw = {"hot": 0, "warm": 1, "cold": 0, "prom_warm": 0, "prom_cold": 0}
    assert analyze_d.rho(uni, allw, True, 128) == pytest.approx((1536 + 1088) / 8192)
