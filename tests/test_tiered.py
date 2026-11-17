"""Tests for the part D simulator (phase0/tiered.py) and its read accounting (analyze_d.py)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def test_fp8_emulation_matches_torch():
    x = torch.randn(50000) * torch.logspace(-4, 2.6, 50000)
    x = x[x.abs() <= 448]
    assert torch.equal(tiered.fp8_e4m3(x), x.to(torch.float8_e4m3fn).float())
    assert tiered.fp8_e4m3(torch.tensor([1000.0, -1000.0])).tolist() == [448.0, -448.0]  # saturates


def test_group256_shares_key_scales_once_group_completes():
    k = torch.randn(H, 600, D).half()
    st = CoarseStore(4, values=False, kcodec="group256")
    st.update(k[:, :200], None)  # 12 blocks: group incomplete, per-block scales
    assert st.groups_done == 0 and not torch.equal(st.k[1][:, 0], st.k[1][:, 1])
    st.update(k, None)  # 37 blocks: groups 0 and 1 complete
    assert st.groups_done == 2
    assert torch.equal(st.k[1][:, 0], st.k[1][:, 15]) and torch.equal(st.k[1][:, 16], st.k[1][:, 31])
    assert not torch.equal(st.k[1][:, 32], st.k[1][:, 33])  # third group still per block
    ref = k[:, :256].float().view(H, 16, BLOCK, D)
    c, s, lo = quantise(ref, 4, dim=(1, 2))
    assert torch.equal(st.k[0][:, :16], c)


def test_static_codec_uses_calibration_and_saturates():
    calib = {"min": torch.full((H, D), -1.0), "max": torch.full((H, D), 1.0),
             "mean": torch.zeros(H, D), "mad": torch.full((H, D), 0.5)}
    k = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0]).repeat(H, 64, D // 5 + 1)[:, :, :D].contiguous().half()
    st = CoarseStore(4, values=False, kcodec="static", calib=calib)
    st.update(k, None)
    dq = st.full("k", k)
    assert dq.max() <= 1.0 + 1e-3 and dq.min() >= -1.0 - 1e-3  # outside the calibrated range saturates
    one = CoarseStore(1, values=False, codec="meanabs", kcodec="static", calib=calib)
    one.update(k, None)
    assert set(one.full("k", k).unique().tolist()) == {-0.5, 0.5}


def test_coldkeys_scores_cold_with_4bit_keys_and_recovers_spike():
    n, spike = 640, 200
    q, k, v = make_inputs(n)
    blk = spike // BLOCK
    u = torch.randn(H, D, generator=torch.Generator().manual_seed(1))
    for h in range(H):
        Q = q[0, h * G : (h + 1) * G, 0].float()
        u[h] -= Q.T @ torch.linalg.solve(Q @ Q.T, Q @ u[h])
        u[h] = 3 * u[h] / u[h].norm() * D**0.5 / 2
        k[0, h, blk * BLOCK : (blk + 1) * BLOCK] = -Q.sum(0).half()
        k[0, h, spike] = u[h].half()
    run(Variant("coldkeys", R=8, eps_w=1e-3, eps_c=0.5), q, k, v, step=0)
    assert (tiered.STATE.layers[0]["plan"][:, blk] == COLD).all()
    assert "scan" not in tiered.STATE.layers[0]  # no separate 1-bit plane
    st = tiered.STATE
    st.step = 1
    st.begin_step()
    tiered.tiered_attention(SimpleNamespace(layer_idx=0), u.repeat_interleave(G, 0).view(1, H * G, 1, D).half(), k, v, None, None)
    assert st.recall[0, 0] >= H and st.recall[0, 1] == st.recall[0, 0]


def test_rho_key_codecs_and_new_kinds():
    allw = {"hot": 0, "warm": 1, "cold": 0, "prom_warm": 0, "prom_cold": 0}
    D = 128
    r = lambda kc: analyze_d.rho({"kind": "uniform", "bits": 4, "R": 1, "kcodec": kc}, allw, True, D)
    assert r("block") == pytest.approx((1024 + 512 + 1088) / 8192)
    assert r("group256") == pytest.approx((1024 + 32 + 1088) / 8192)
    assert r("static") == pytest.approx((1024 + 1088) / 8192)
    assert analyze_d.rho({"kind": "fp8", "bits": 8, "R": 1}, allw, True, D) == 0.5
    fr = {"hot": 0.1, "warm": 0.5, "cold": 0.4, "prom_warm": 0.02, "prom_cold": 0.01}
    e = analyze_d.rho({"kind": "coldkeys", "bits": 4, "R": 4}, fr, False, D)
    assert e == pytest.approx(0.1 + 0.48 * 0.25 + 0.02 * 1.125 + 0.4 * 0.125 + 0.01)


def _row(name, kind, ppl, rho_meta, bits=4, kc="block", hot=16):
    return {"variant": name, "kind": kind, "bits": bits, "kcodec": kc, "hot_bits": hot,
            "ppl_increase": ppl, "ppl_increase_ci_hi": ppl + 0.004, "rho_meta": rho_meta}


def test_decision_rule_cross_codec_bar():
    base = [_row("fp16", "fp16", 0.0, 1.0, bits=16, kc=""), _row("fp8", "fp8", 0.0, 0.5, bits=8, kc=""),
            _row("u8", "uniform", 0.0, 0.57, 8), _row("u4", "uniform", 0.002, 0.32, 4),
            _row("u4g", "uniform", 0.006, 0.264, 4, kc="group256"), _row("u3g", "uniform", 0.03, 0.2, 3, kc="group256"),
            _row("u8s", "uniform", 0.0, 0.51, 8, kc="static"), _row("u4s", "uniform", 0.02, 0.26, 4, kc="static")]
    d = analyze_d.decide(base)
    assert d["bar"]["variant"] == "u4g" and d["rho_meta_budget"] == pytest.approx(0.198)  # cheapest passing, across codecs
    assert d["verdict"].startswith("incomplete") and d["per_codec"]["static"]["b_star"] == 8
    assert "fp8" not in json.dumps(d["bar"])  # fp8 (0.5, passing) is excluded
    go = analyze_d.decide(base + [_row("e", "coldkeys", 0.009, 0.19, bits=2, kc="group256", hot=4)])
    assert go["verdict"] == "go" and go["passing"][0]["variant"] == "e"
    over = analyze_d.decide(base + [_row("f", "live", 0.005, 0.22, bits=3, kc="group256", hot=4)])
    assert over["verdict"] == "no-go so far: tiered variants within 1% read more than the budget"
    bad = analyze_d.decide(base + [_row("f", "live", 0.05, 0.15, bits=2, kc="group256", hot=4)])
    assert bad["verdict"] == "no-go so far: every tiered variant fails on quality"


def test_rho_with_hot_at_bstar():
    fr = {"hot": 0.1, "warm": 0.5, "cold": 0.4, "prom_warm": 0.02, "prom_cold": 0.01}
    D = 128
    k4, v4 = (1024 + 512) / 8192, (1024 + 64) / 8192
    h = k4 + v4
    k3 = (768 + 512) / 8192
    v3 = (768 + 64) / 8192
    got = analyze_d.rho({"kind": "coldkeys", "bits": 3, "R": 4, "hot_bits": 4}, fr, True, D)
    assert got == pytest.approx(0.1 * h + 0.48 * (k3 + v3) + 0.02 * (k3 + h) + 0.4 * k3 + 0.01 * h)


def test_hot_at_bstar_reads_hot_blocks_from_bstar_copy():
    q, k, v = make_inputs(320)
    a = run(Variant("fresh", eps_w=-1.0, eps_c=0.0, hot_bits=4), q, k, v)  # everything HOT, at 4 bits
    b = run(Variant("uniform", bits=4), q, k, v)
    assert torch.allclose(a, b, atol=1e-4)
    assert Variant("coldkeys", R=4, eps_w=1e-3, eps_c=0.02, bits=3, hot_bits=4, kcodec="group256").name == \
        "coldkeys_R4_w0.001_c0.02_h4b3_kgroup256"


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    nf = rng.uniform(200, 400, 40)
    nv = nf * (1 + rng.normal(0.01, 0.01, 40))
    t = np.full(40, 128)
    lo, hi = analyze_d.bootstrap_ppl_ci(nv, nf, t)
    point = np.expm1((nv.sum() - nf.sum()) / t.sum())
    assert lo < point < hi


def test_asymmetric_uniform_names_store_and_rho():
    vs = tiered.build_variants(["uniformK4V3"], [1e-3], [0.02], 4, kcodecs=["group256"])
    assert [v.name for v in vs] == ["uniformK4V3_kgroup256"] and vs[0].value_bits == 3
    assert Variant("uniform", bits=4, vbits=4).name == "uniform4"
    q, k, v = make_inputs(320)
    st = CoarseStore(4, values=True, vbits=2)
    st.update(k[0], v[0])
    assert st.k[0].max() == 15 and st.v[0].max() <= 3
    a = run(Variant("uniform", bits=4, vbits=2), q, k, v)
    b = run(Variant("uniform", bits=4), q, k, v)
    assert not torch.allclose(a, b)
    allw = {"hot": 0, "warm": 1, "cold": 0, "prom_warm": 0, "prom_cold": 0}
    D = 128
    got = analyze_d.rho({"kind": "uniform", "bits": 4, "vbits": 3, "R": 1, "kcodec": "group256"}, allw, True, D)
    assert got == pytest.approx((1024 + 32 + 768 + 64) / 8192)


def test_asymmetric_uniform_can_set_the_bar_but_not_b_star():
    base = [_row("fp16", "fp16", 0.0, 1.0, bits=16, kc=""), _row("u4g", "uniform", 0.006, 0.264, 4, kc="group256"),
            _row("u3g", "uniform", 0.03, 0.2, 3, kc="group256"),
            _row("k4v3", "uniform", 0.008, 0.233, 4, kc="group256") | {"vbits": 3}]
    d = analyze_d.decide(base)
    assert d["bar"]["variant"] == "k4v3" and d["rho_meta_budget"] == pytest.approx(0.75 * 0.233)
    assert d["per_codec"]["group256"]["b_star"] == 4 and d["per_codec"]["group256"]["uniform_b_star_rho_meta"] == 0.264
    old = analyze_d.decide(base[:3] + [_row("c", "promote", 0.001, 0.1, bits=3, kc="group256", hot=4)])
    assert old["verdict"].startswith("incomplete") and not old["tiered"][0]["decides"]  # c no longer decides


def test_live_all_hot_is_exact_and_uses_no_stored_plan():
    q, k, v = make_inputs(320)
    o = run(Variant("live", bits=3, eps_w=-1.0, eps_c=0.0), q, k, v)
    assert torch.allclose(o, exact(q, k, v), atol=1e-4)
    assert "plan" not in tiered.STATE.layers[0]


def test_live_finds_spike_from_warm_keys_and_skips_cold_values():
    n, spike = 640, 300
    q, k, v = make_inputs(n, spike=spike)
    o = run(Variant("live", bits=4, eps_w=1e-3, eps_c=0.5), q, k, v)
    st = tiered.STATE
    c = st.counts[0]
    assert c[2] > 0 and c[3] == 0 and c[4] == 0  # some COLD, no fix-ups
    ev, hit, live_hot, miss_cold = st.recall[0].tolist()
    assert ev > 0 and hit >= 0.9 * ev and miss_cold <= 0.1 * ev  # live plan mostly agrees with the oracle
    assert ((o - exact(q, k, v)).norm() / exact(q, k, v).norm()) < 0.1
    assert Variant("live", bits=3, eps_w=1e-3, eps_c=0.02, hot_bits=4, kcodec="group256").name == \
        "live_w0.001_c0.02_h4b3_kgroup256"


def test_rho_live():
    fr = {"hot": 0.1, "warm": 0.5, "cold": 0.4, "prom_warm": 0.0, "prom_cold": 0.0}
    D = 128
    k4, v4 = (1024 + 32) / 8192, (1024 + 64) / 8192
    k3, v3 = (768 + 32) / 8192, (768 + 64) / 8192
    got = analyze_d.rho({"kind": "live", "bits": 3, "R": 1, "hot_bits": 4, "kcodec": "group256"}, fr, True, D)
    assert got == pytest.approx(0.1 * (k3 + k4 + v4) + 0.5 * (k3 + v3) + 0.4 * k3)


def test_by_length_groups_prompts():
    rows = [{"variant": "e", "prompt_len": 8193, "tokens": 128, "hot": 0.1, "warm": 0.5, "cold": 0.4,
             "nll_sum": 101.0, "fp16_nll_sum": 100.0},
            {"variant": "e", "prompt_len": 8190, "tokens": 128, "hot": 0.3, "warm": 0.5, "cold": 0.2,
             "nll_sum": 100.0, "fp16_nll_sum": 100.0, "task_score": 1.0},
            {"variant": "e", "prompt_len": 32001, "tokens": 128, "hot": 0.05, "warm": 0.3, "cold": 0.65,
             "nll_sum": 100.0, "fp16_nll_sum": 100.0}]
    out = analyze_d.by_length(rows, {"e": "coldkeys"})
    assert [(r["context_k"], r["prompts"]) for r in out] == [(8, 2), (31, 1)]
    assert out[0]["hot"] == pytest.approx(0.2) and out[0]["task_score"] == 1.0
    assert out[0]["ppl_increase"] == pytest.approx(np.expm1(1 / 256))


def test_decision_comparison_rows():
    base = [_row("fp16", "fp16", 0.0, 1.0, bits=16, kc=""), _row("u4", "uniform", 0.002, 0.32, 4),
            _row("u4g", "uniform", 0.002, 0.27, 4, kc="group256"),
            {**_row("k4v3", "uniform", 0.003, 0.24, 4, kc="group256"), "vbits": 3},
            _row("e", "coldkeys", 0.002, 0.16, bits=3, kc="group256", hot=4),
            _row("f", "live", 0.004, 0.16, bits=3, kc="group256", hot=4)]
    for r in base:
        r.update(worst_prompt=f"{r['variant']}-p", ppl_increase_prompt_max=0.03, prompts_above_2pct=2, ppl_increase_ci_lo=0.0)
    d = analyze_d.decide(base)
    roles = {c["role"]: c["variant"] for c in d["comparison"]}
    assert roles == {"bar": "k4v3", "uniform4": "u4g", "coldkeys": "e", "live": "f"}  # group256 4-bit, not the asymmetric bar
    assert all(c["prompts_above_2pct"] == 2 and c["worst_prompt"].endswith("-p") for c in d["comparison"])


def test_prompt_flags():
    rows = [{"variant": "a", "prompt": "p1", "ppl_increase": 0.01, "nll_sum": 1.0, "fp16_nll_sum": 1.0, "tf_agree": 0.5},
            {"variant": "a", "prompt": "p2", "ppl_increase": -0.12, "nll_sum": 1.0, "fp16_nll_sum": 1.1, "tf_agree": 0.5},
            {"variant": "a", "prompt": "p3", "ppl_increase": float("nan"), "nll_sum": float("nan"), "fp16_nll_sum": 1.0, "tf_agree": 0.5},
            {"variant": "a", "prompt": "p4", "ppl_increase": "", "nll_sum": 1.0, "fp16_nll_sum": float("inf"), "tf_agree": 0.5}]
    f = {r["prompt"]: r["reason"] for r in analyze_d.prompt_flags(rows)}
    assert set(f) == {"p2", "p3", "p4"}
    assert "> 10%" in f["p2"] and "non-finite nll_sum" in f["p3"] and "non-finite fp16_nll_sum" in f["p4"]


def test_check_versions_refuses_mixed(tmp_path):
    m = tmp_path / "meta.json"
    tiered.check_versions(m, {"torch": "2.15.0", "transformers": "4.57.3"}, False)  # no meta yet: fine
    m.write_text(json.dumps({"versions": {"torch": "2.9.0", "transformers": "4.57.3"},
                             "invocations": [{"versions": {"torch": "2.9.0", "transformers": "4.57.3"}}]}))
    tiered.check_versions(m, {"torch": "2.9.0", "transformers": "4.57.3"}, False)  # same: fine
    with pytest.raises(SystemExit):
        tiered.check_versions(m, {"torch": "2.15.0", "transformers": "4.57.3"}, False)
    tiered.check_versions(m, {"torch": "2.15.0", "transformers": "4.57.3"}, True)  # explicitly allowed
