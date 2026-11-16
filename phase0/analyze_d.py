#!/usr/bin/env python
"""Phase 0, part D analysis: quality against read fraction for each tiered variant.

Reads the output directories of tiered.py and writes, per variant:

    ppl_increase      pooled teacher-forced perplexity over all prompts, relative to fp16
                      (exp of mean NLL over every reference token), plus the per-prompt
                      median and max increase
    tf_agree          teacher-forced argmax agreement with the reference tokens
    attn_err          relative L2 error of each layer's attention output against exact
                      attention on the same inputs (mean over layers; per layer in
                      per_layer_err.csv)
    free_agree        free-running greedy: fraction of positions matching the fp16 greedy
                      output, and the matching prefix length (with tiered.py --free-run)
    task_score        benchmark score of the free-running output where a metric exists
    reads             block fractions: planned HOT / WARM / COLD, promoted, effective HOT
    fixup_rate        blocks read at fp16 in the second pass per step, as a fraction of all
                      blocks, and per KV head per step
    recall_cold/warm  of the blocks HOT under the fresh oracle plan but COLD (WARM) under
                      the stale plan, the fraction the same-step scan promoted
    promotion_precision  of the promoted blocks, the fraction HOT under the fresh oracle plan
    tail              per query head true attention mass in COLD blocks, before and after
                      promotion: mean and fraction of query heads above 0.02, 0.05, 0.10
    rho_codes         read fraction counting code bits only (PHASE0 E, extended with scan
                      and fix-up reads)
    rho_meta          the same, adding quantiser metadata: fp16 scale and offset per channel
                      for keys (per block, per 256-token group, or none for static) and per
                      token for values
    ppl_increase_ci   95% bootstrap interval over prompts

Read accounting, as fractions of an fp16 block (keys and values):
    fp16 read 1; coarse read k(c) + v(c); coarse key only k(c); 1-bit key scan k(1)
    codes only: k(b) = v(b) = b/32
    with metadata (head dim D): k(b) = (2bD + 4D/S) / 64D with S blocks sharing a key scale
    (1 block, 16 for group256, static: no per-block term), v(b) = (2bD + 64) / 64D
    fp8         0.5
    uniform     HOT + WARM (k+v)
    hot_only    HOT
    fresh       HOT + WARM (k+v) + COLD k(c)          (refresh every step)
    stale       HOT + WARM (k+v) + COLD k(c) / R
    promote     HOT + (WARM - PW)(k+v) + PW (k(c) + 1) + COLD k(1) + PC + COLD k(c) / R
    twotier     HOT + (WARM - PW)(k+v) + PW (k(c) + 1)
    coldkeys    HOT + (WARM - PW)(k+v) + PW (k(c) + 1) + COLD k(c) + PC
where PW and PC are blocks promoted from WARM and COLD. The COLD k(c)/R term charges the
periodic refresh as a coarse key scan (the simulation refreshes from exact weights).

Decision (docs/PHASE0.md), written to decision.json: per key codec, b* is the lowest
uniform width within 1% perplexity; the ladder is HOT = b*, WARM = the next tested width
below b*, COLD as before. Go if a variant c, d or e on that ladder stays within 1% and its
rho_meta is at most 0.75 x that of uniform b*. No-go if no ladder below b* passes. Point
estimates decide; rows whose 95% CI upper bound crosses 1% are flagged. ppl_increase is
paired: over the prompts a variant ran on, against fp16 on the same prompts. fp8 is an
emulation and is labelled "emulation only"; it is not a candidate for b*.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

PPL_LIMIT = 0.01  # perplexity increase allowed for "stays within 1%"
MIN_SAVING = 0.25  # tiered must read at least 25% fewer bytes than the cheapest passing uniform cache
TIERED_GO_KINDS = ("promote", "twotier", "coldkeys")  # variants c, d, e
KEY_META_BLOCKS = {"block": 1, "group256": 16, "static": None}  # blocks sharing one key scale/offset
BOOTSTRAP = 2000
TAIL = (0.02, 0.05, 0.10)


def costs(with_meta: bool, D: int, kcodec: str = "block"):
    """Bytes of a b-bit key / value read for one block and KV head, as a fraction of the
    fp16 block (64 D bytes for keys and values). Metadata: an fp16 scale and offset per
    channel (4 D bytes) shared by KEY_META_BLOCKS blocks for keys, per token for values."""
    if not with_meta:
        return (lambda b: b / 32), (lambda b: b / 32)
    shared = KEY_META_BLOCKS[kcodec]
    kmeta = 0.0 if shared is None else 4 * D / shared
    return (lambda b: (2 * b * D + kmeta) / (64 * D)), (lambda b: (2 * b * D + 64) / (64 * D))


def rho(v: dict, fr: dict, with_meta: bool, D: int) -> float:
    k, val = costs(with_meta, D, v.get("kcodec", "block"))
    kind, c, R, s = v["kind"], v["bits"], v["R"], v.get("scan_bits", 1)
    hb = v.get("hot_bits", 16)
    h = 1.0 if hb >= 16 else k(hb) + val(hb)  # cost of a HOT (or fix-up) read
    H, W, C, PW, PC = fr["hot"], fr["warm"], fr["cold"], fr["prom_warm"], fr["prom_cold"]
    if kind == "fp16":
        return 1.0
    if kind == "fp8":
        return 0.5  # per-tensor scale (1.0); no per-block metadata
    if kind == "uniform":
        return H + W * (k(c) + val(c))
    if kind == "hot_only":
        return H * h
    if kind == "fresh":
        return H * h + W * (k(c) + val(c)) + C * k(c)
    if kind == "stale":
        return H * h + W * (k(c) + val(c)) + C * k(c) / R
    if kind == "promote":
        return H * h + (W - PW) * (k(c) + val(c)) + PW * (k(c) + h) + C * k(s) + PC * h + C * k(c) / R
    if kind == "twotier":
        return H * h + (W - PW) * (k(c) + val(c)) + PW * (k(c) + h)
    if kind == "coldkeys":  # COLD keys read at c bits every step, so no separate refresh scan
        return H * h + (W - PW) * (k(c) + val(c)) + PW * (k(c) + h) + C * k(c) + PC * h
    raise ValueError(kind)


def bootstrap_ppl_ci(nll_v, nll_f, tokens, n=BOOTSTRAP, seed=0):
    """95% CI of the pooled perplexity increase, resampling prompts (tokens within a prompt
    are correlated, so prompts are the unit)."""
    nll_v, nll_f, tokens = map(np.asarray, (nll_v, nll_f, tokens))
    if len(tokens) < 2:
        return float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(tokens), size=(n, len(tokens)))
    delta = (nll_v[idx].sum(1) - nll_f[idx].sum(1)) / tokens[idx].sum(1)
    lo, hi = np.percentile(np.expm1(delta), [2.5, 97.5])
    return float(lo), float(hi)


WIDTHS = (16, 8, 4, 3, 2)  # uniform widths; 16 is fp16


def decide(rows: list[dict]) -> dict:
    """Go/no-go rule of docs/PHASE0.md on summary rows (paired point estimates).

    Per key codec: b* is the lowest uniform width within 1% (fp16 counts as 16 and always
    qualifies); the ladder is HOT = b*, WARM = the next tested width below b*. Go if a
    variant c/d/e on that ladder and codec stays within 1% and reads at least 25% fewer
    bytes (rho_meta) than uniform b* with the same codec. fp8 is emulation only and not
    a candidate for b*.
    """
    ok = lambda r: r["ppl_increase"] <= PPL_LIMIT
    fp16 = next((r for r in rows if r["kind"] == "fp16"), None)
    codecs = sorted({r["kcodec"] for r in rows if r["kind"] == "uniform"})
    per_codec, any_ladder, missing, passing_all = {}, False, [], []
    for kc in codecs:
        uni = {r["bits"]: r for r in rows if r["kind"] == "uniform" and r["kcodec"] == kc}
        if fp16:
            uni[16] = fp16
        tested = [b for b in WIDTHS if b in uni]
        passing = [b for b in tested if ok(uni[b])]
        bstar = min(passing) if passing else None
        below = [b for b in tested if bstar and b < bstar]
        warm = max(below) if below else None
        non_monotone = [b for b in tested if bstar and b > bstar and not ok(uni[b])]
        entry = {"b_star": bstar, "warm_bits": warm, "uniform_b_star_rho_meta": uni[bstar]["rho_meta"] if bstar else None,
                 "uniform_widths_within_limit": passing, "non_monotone_widths": non_monotone}
        if bstar and warm:
            any_ladder = True
            budget = (1 - MIN_SAVING) * uni[bstar]["rho_meta"]
            ladder = [r for r in rows if r["kind"] in TIERED_GO_KINDS and r["kcodec"] == kc
                      and r.get("hot_bits", 16) == bstar and r["bits"] == warm]
            good = [r for r in ladder if ok(r) and r["rho_meta"] <= budget]
            entry.update(rho_meta_budget=budget, ladder_variants_run=[r["variant"] for r in ladder],
                         passing=[{"variant": r["variant"], "rho_meta": r["rho_meta"], "ppl_increase": r["ppl_increase"],
                                   "ci_hi_above_limit": r["ppl_increase_ci_hi"] > PPL_LIMIT} for r in good])
            if not ladder:
                missing.append(kc)
            passing_all += good
        per_codec[kc] = entry
    if not codecs:
        verdict = "incomplete: no uniform runs"
    elif passing_all:
        verdict = "go"
    elif not any_ladder:
        verdict = "no-go: b* is the lowest tested width for every key codec, so no ladder exists below it"
    elif missing:
        verdict = f"incomplete: ladder variants not run for key codec(s) {missing}"
    else:
        verdict = "no-go: no ladder below b* passes"
    return {
        "verdict": verdict,
        "per_codec": per_codec,
        "rule": f"per key codec, b* = lowest uniform width (16/8/4/3/2) with ppl increase <= {PPL_LIMIT:.0%}; ladder HOT=b*, "
                f"WARM=next tested width below b*; go if a c/d/e variant on the ladder has ppl increase <= {PPL_LIMIT:.0%} "
                f"and rho_meta <= {1 - MIN_SAVING:.2f} x uniform b*; fp8 is emulation only and excluded",
    }


def free_agreement(gen, ref):
    n = min(len(gen), len(ref))
    match = sum(a == b for a, b in zip(gen, ref))
    prefix = next((i for i in range(n) if gen[i] != ref[i]), n)
    return match / max(len(ref), 1), prefix


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True, help="tiered.py output directories")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokenizer", default=None, help="for decoding free-running output to score tasks (default: the run's model)")
    args = p.parse_args()

    metas = {str(r): json.loads((r / "meta.json").read_text()) for r in args.runs}
    models = {m["model"] for m in metas.values()}
    if len(models) != 1:
        sys.exit(f"runs mix models: {models}")
    shape = next(iter(metas.values()))["model_shape"]
    L, Hq, D = shape["layers"], shape["q_heads"], shape["head_dim"]
    vdefs = {}
    for m in metas.values():
        for v in m["variants"]:
            vdefs[v["name"]] = v

    tok = None
    acc = defaultdict(lambda: defaultdict(float))
    layer_num = defaultdict(lambda: np.zeros(L))
    layer_den = defaultdict(lambda: np.zeros(L))
    per_prompt = []
    n_prompts = 0
    for run in args.runs:
        ids = dict.fromkeys(json.loads(l)["id"] for l in (run / "done.jsonl").read_text().splitlines() if l.strip())
        for pid in ids:  # a prompt appears once per tiered.py invocation that added variants to it
            pdir = run / pid
            info = json.loads((pdir / "prompt.json").read_text())
            n_prompts += 1
            data = {}
            for f in sorted(pdir.glob("*.npz")):
                data[f.stem] = np.load(f)
            if "fp16" not in data:
                sys.exit(f"{pdir}: no fp16 result")
            fp_nll = data["fp16"]["nll"]
            for name, d in data.items():
                a = acc[name]
                T = len(d["nll"])
                a["tokens"] += T
                a["nll"] += float(d["nll"].sum())
                a["agree"] += float(d["agree"].sum())
                if T:
                    layer_num[name] += d["err_num"].sum(0)
                    layer_den[name] += d["err_den"].sum(0)
                    c = d["counts"].sum((0, 1))
                    for i, k in enumerate(("hot", "warm", "cold", "prom_warm", "prom_cold", "total")):
                        a[k] += float(c[i])
                    r = d["recall"].sum((0, 1))
                    for i, k in enumerate(("ev_cold", "rec_cold", "ev_warm", "rec_warm")):
                        a[k] += float(r[i])
                    t = d["tail"].sum((0, 1))
                    for i, k in enumerate(["before_sum"] + [f"before_gt{x}" for x in TAIL] + ["after_sum"] + [f"after_gt{x}" for x in TAIL]):
                        a[k] += float(t[i])
                    a["qhead_steps"] += T * L * Hq
                    a["layer_steps"] += T * L
                ppl_inc = math.exp(d["nll"].mean() - fp_nll.mean()) - 1 if T else float("nan")
                row = {"run": str(run), "prompt": pid, "task": info.get("task"), "variant": name, "tokens": T,
                       "nll_sum": float(d["nll"].sum()), "fp16_nll_sum": float(fp_nll.sum()),
                       "ppl_increase": ppl_inc, "tf_agree": float(d["agree"].mean()) if T else float("nan")}
                if "free_ids" in d.files:
                    ref = data["fp16"]["free_ids"].tolist()
                    fa, prefix = free_agreement(d["free_ids"].tolist(), ref)
                    row.update(free_agree=fa, free_prefix=prefix)
                    a["free_agree"] += fa
                    a["free_n"] += 1
                    if info.get("metric") and info.get("answers"):
                        if tok is None:
                            from transformers import AutoTokenizer
                            tok = AutoTokenizer.from_pretrained(args.tokenizer or next(iter(models)))
                        from benchmarks import score
                        ids = d["free_ids"].tolist()
                        eos = set(metas[str(run)]["eos_ids"])
                        ids = ids[: next((i for i, x in enumerate(ids) if x in eos), len(ids))]
                        pred = tok.decode(ids, skip_special_tokens=True)
                        row["task_score"] = score(info["metric"], pred, info["answers"], info.get("all_classes"), info.get("task"))
                        a["task_score"] += row["task_score"]
                        a["task_n"] += 1
                per_prompt.append(row)

    fp = acc["fp16"]
    fp_ppl = math.exp(fp["nll"] / fp["tokens"])
    rows, layer_rows = [], []
    for name, a in acc.items():
        v = vdefs[name]
        tot = max(a["total"], 1)
        fr = {k: a[k] / tot for k in ("hot", "warm", "cold", "prom_warm", "prom_cold")}
        pp = [r["ppl_increase"] for r in per_prompt if r["variant"] == name and not math.isnan(r["ppl_increase"])]
        err_layers = np.sqrt(layer_num[name] / np.maximum(layer_den[name], 1e-30))
        hot_eff = fr["hot"] + fr["prom_warm"] + fr["prom_cold"]
        mine = [r for r in per_prompt if r["variant"] == name and r["tokens"]]
        ci = bootstrap_ppl_ci([r["nll_sum"] for r in mine], [r["fp16_nll_sum"] for r in mine], [r["tokens"] for r in mine])
        paired = math.expm1((sum(r["nll_sum"] for r in mine) - sum(r["fp16_nll_sum"] for r in mine))
                            / max(sum(r["tokens"] for r in mine), 1))
        row = {
            "variant": name, "kind": v["kind"], "bits": v["bits"] if v["kind"] in ("uniform",) + TIERED_GO_KINDS + ("fresh", "stale") else "",
            "kcodec": v.get("kcodec", "block") if v["kind"] not in ("fp16", "fp8", "hot_only") else "",
            "hot_bits": v.get("hot_bits", 16) if v["kind"] in TIERED_GO_KINDS + ("fresh", "stale", "hot_only") else "",
            "note": "emulation only" if v["kind"] == "fp8" else "",
            "R": v["R"] if v["kind"] in ("stale",) + TIERED_GO_KINDS else "",
            "eps_w": v["eps_w"] if v["kind"] not in ("fp16", "fp8", "uniform") else "",
            "eps_c": v["eps_c"] if v["kind"] in ("fresh", "stale", "promote", "coldkeys") else "",
            "scan_codec": v.get("scan_codec") if v["kind"] == "promote" else "",
            "tokens": int(a["tokens"]),
            "ppl": math.exp(a["nll"] / a["tokens"]),
            "prompts": len(mine),
            "ppl_increase": paired,  # over the prompts this variant ran on, against fp16 on the same prompts
            "ppl_increase_ci_lo": ci[0], "ppl_increase_ci_hi": ci[1],
            "ppl_increase_prompt_median": float(np.median(pp)) if pp else float("nan"),
            "ppl_increase_prompt_max": max(pp) if pp else float("nan"),
            "tf_agree": a["agree"] / a["tokens"],
            "attn_err_mean": float(err_layers.mean()),
            "attn_err_max_layer": float(err_layers.max()),
            "free_agree": a["free_agree"] / a["free_n"] if a["free_n"] else "",
            "task_score": a["task_score"] / a["task_n"] if a["task_n"] else "",
            "hot_planned": fr["hot"], "warm": fr["warm"], "cold": fr["cold"],
            "promoted_warm": fr["prom_warm"], "promoted_cold": fr["prom_cold"], "hot_effective": hot_eff,
            "fixup_rate": fr["prom_warm"] + fr["prom_cold"],
            "fixups_per_kv_head_step": (a["prom_warm"] + a["prom_cold"]) / max(a["layer_steps"], 1) / shape["kv_heads"],
            "recall_cold": a["rec_cold"] / a["ev_cold"] if a["ev_cold"] else "",
            "recall_warm": a["rec_warm"] / a["ev_warm"] if a["ev_warm"] else "",
            "promotion_precision": (a["rec_cold"] + a["rec_warm"]) / (a["prom_warm"] + a["prom_cold"]) if a["prom_warm"] + a["prom_cold"] else "",
            "stale_cold_events": int(a["ev_cold"]),
            "cold_mass_before": a["before_sum"] / max(a["qhead_steps"], 1),
            "cold_mass_after": a["after_sum"] / max(a["qhead_steps"], 1),
            **{f"frac_qheads_cold_mass_gt_{x}_before": a[f"before_gt{x}"] / max(a["qhead_steps"], 1) for x in TAIL},
            **{f"frac_qheads_cold_mass_gt_{x}_after": a[f"after_gt{x}"] / max(a["qhead_steps"], 1) for x in TAIL},
            "rho_codes": rho(v, fr, False, D),
            "rho_meta": rho(v, fr, True, D),
        }
        rows.append(row)
        layer_rows += [{"variant": name, "layer": i, "attn_err": float(e)} for i, e in enumerate(err_layers)]
    order = ["fp16", "fp8", "uniform", "hot_only", "fresh", "stale", "promote", "twotier", "coldkeys"]
    rows.sort(key=lambda r: (order.index(r["kind"]), str(r["R"]), r["variant"]))

    args.out.mkdir(parents=True, exist_ok=True)

    def write(name, rs):
        keys = list(dict.fromkeys(k for r in rs for k in r))
        with open(args.out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rs:
                w.writerow({k: (f"{x:.6g}" if isinstance(x, float) else x) for k, x in r.items()})

    write("summary.csv", rows)
    write("per_layer_err.csv", layer_rows)
    write("per_prompt.csv", per_prompt)
    decision = decide(rows)
    (args.out / "decision.json").write_text(json.dumps(decision, indent=2))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        commit = None
    (args.out / "meta.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "runs": [str(r) for r in args.runs], "prompts": n_prompts,
        "git_commit": commit, "decision": decision, "bootstrap_resamples": BOOTSTRAP, "fp16_ppl": fp_ppl, "run_meta": metas}, indent=2))

    print(f"{n_prompts} prompts, {int(fp['tokens'])} reference tokens per variant, fp16 ppl {fp_ppl:.4f} -> {args.out}")
    hdr = f"{'variant':44} {'dPPL% [95% CI]':>22} {'agree':>6} {'aErr':>6} {'rho':>6} {'rhoM':>6} {'HOTeff':>6} {'fixup':>6} {'recC':>5} {'prec':>5} {'tail>.05 b/a':>13}"
    print(hdr)
    for r in rows:
        rc = r["recall_cold"]
        ci = f"{100 * r['ppl_increase']:6.2f} [{100 * r['ppl_increase_ci_lo']:5.2f},{100 * r['ppl_increase_ci_hi']:5.2f}]"
        label = r["variant"] + (" (emulation only)" if r["note"] else "")
        print(f"{label:44} {ci:>22} {r['tf_agree']:6.3f} {r['attn_err_mean']:6.4f} {r['rho_codes']:6.3f} "
              f"{r['rho_meta']:6.3f} {r['hot_effective']:6.3f} {r['fixup_rate']:6.4f} {rc if rc == '' else format(rc, '.2f'):>5} "
              f"{r['promotion_precision'] if r['promotion_precision'] == '' else format(r['promotion_precision'], '.2f'):>5} "
              f"{r['frac_qheads_cold_mass_gt_0.05_before']:6.4f}/{r['frac_qheads_cold_mass_gt_0.05_after']:.4f}")
    print(f"\nDecision: {decision['verdict']}")
    for kc, e in decision["per_codec"].items():
        print(f"  {kc:9} b*={e['b_star']} warm={e['warm_bits']} uniform b* rho_meta={e['uniform_b_star_rho_meta']}"
              + (f" budget<={e['rho_meta_budget']:.3f} ladder run={len(e['ladder_variants_run'])} passing={[x['variant'] for x in e['passing']]}"
                 if e.get("rho_meta_budget") else "")
              + (f" NON-MONOTONE {e['non_monotone_widths']}" if e["non_monotone_widths"] else ""))


if __name__ == "__main__":
    main()
