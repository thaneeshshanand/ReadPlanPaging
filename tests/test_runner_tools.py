"""Tests for the replication check (compare_runs.py), the timing estimate (estimate_time.py)
and the prompt selection used by the runner's per-length steps (tiered.select_prompts)."""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase0"))
import compare_runs  # noqa: E402
import estimate_time  # noqa: E402
from tiered import select_prompts  # noqa: E402


def test_select_prompts_filters_length_before_limit():
    recs = [{"id": f"p{i}", "n_tokens": L} for i, L in enumerate([8192, 16384, 32000] * 3)]
    assert [r["id"] for r in select_prompts(recs, [16384], limit=1)] == ["p1"]
    assert [r["id"] for r in select_prompts(recs, [32000])] == ["p2", "p5", "p8"]
    assert len(select_prompts(recs)) == 9


def _row(v, p, diff_pp):
    return {"variant": v, "prompt": p, "increase_diff_pp": diff_pp}


def _dirs(tmp_path, va, vb):
    for k, v in (("a", va), ("b", vb)):
        (tmp_path / k).mkdir()
        (tmp_path / k / "decision.json").write_text(json.dumps({"verdict": v}))
    return tmp_path / "a", tmp_path / "b"


def test_replication_check_pass_and_each_failure(tmp_path):
    a, b = _dirs(tmp_path, "go", "go")
    sa = {"e": {"ppl_increase": "0.0030"}}
    sb = {"e": {"ppl_increase": "0.0010"}}  # 0.20 points apart
    rows = [_row("e", "p1", 1.9), _row("e", "p2", -4.9)]
    chk = compare_runs.replication_check(a, b, sa, sb, ["e"], rows, 0.25, 5.0)
    assert chk["pass"], chk
    assert not compare_runs.replication_check(a, b, {"e": {"ppl_increase": "0.0040"}}, sb, ["e"], rows, 0.25, 5.0)["pass"]
    assert not compare_runs.replication_check(a, b, sa, sb, ["e"], rows + [_row("e", "p3", 5.1)], 0.25, 5.0)["pass"]
    assert not compare_runs.replication_check(a, b, sa, sb, ["e"], rows + [_row("e", "p3", math.nan)], 0.25, 5.0)["pass"]
    (tmp_path / "x").mkdir()
    a2, b2 = _dirs(tmp_path / "x", "go", "no-go so far: x")
    assert not compare_runs.replication_check(a2, b2, sa, sb, ["e"], rows, 0.25, 5.0)["pass"]


def test_drift_table_quantiles_and_bands():
    rows = [_row("e", f"p{i}", d) for i, d in enumerate([0.1, -0.2, 0.5, 1.5, -3.0, 6.0])]
    (d,) = compare_runs.drift_table(rows, ["e"])
    assert d["n"] == 6 and d["max"] == 6.0 and math.isclose(d["median"], 1.0)
    assert [d[b] for b in compare_runs.BAND_NAMES] == [2, 1, 1, 1, 1]


def test_estimate_counts_remaining_work(tmp_path):
    t = tmp_path / "timing"
    t.mkdir()
    done = [{"id": "a-8192", "prompt_len": 8192, "timings": {"prefill_s": 10, "fp16": 20, "uniform4_kgroup256": 30}},
            {"id": "a-8192", "prompt_len": 8192, "timings": {"prefill_s": 10, "coldkeys_R4_w0.001_c0.02_h4b3_kgroup256": 60,
                                                             "live_w0.001_c0.02_h4b3_kgroup256": 50}}]
    (t / "done.jsonl").write_text("".join(json.dumps(r) + "\n" for r in done))
    recs = [{"id": "a-8192", "n_tokens": 8192}, {"id": "b-8192", "n_tokens": 8192}, {"id": "c-16384", "n_tokens": 16384}]
    tim = estimate_time.load_timings(t, {r["id"]: r["n_tokens"] for r in recs})
    assert tim[8192] == {"prefill_s": 10, "fp16": 20, "uniform4": 30, "coldkeys_R4": 60, "live": 50}
    out = tmp_path / "out"
    (out / "b-8192").mkdir(parents=True)
    (out / "b-8192" / "coldkeys_R4_w0.001_c0.02_h4b3_kgroup256.npz").write_bytes(b"")
    (out / "b-8192" / "live_w0.001_c0.02_h4b3_kgroup256.npz").write_bytes(b"")
    est = estimate_time.estimate(tim, recs, out)
    # uniform: 2 prompts x (10 + 20 + 5 x 30) s; 16K not measured
    assert math.isclose(est["uniform"]["by_length"][8192]["hours"], 2 * 180 / 3600)
    assert not est["uniform"]["complete"]
    # ladder: a needs 4/3 and 4/2, b only 4/2: 3 passes x (10 + 60 + 50) s
    assert est["ladder"]["by_length"][8192]["passes_to_do"] == 3
    assert math.isclose(est["ladder"]["by_length"][8192]["hours"], 3 * 120 / 3600)
