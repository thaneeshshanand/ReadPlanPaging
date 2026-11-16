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
    rho_meta          the same, adding per-block quantiser metadata: fp16 scale and offset
                      per channel per block for keys, per token for values

Read accounting, as fractions of an fp16 block (keys and values):
    fp16 read 1; coarse read k(c) + v(c); coarse key only k(c); 1-bit key scan k(1)
    codes only: k(b) = v(b) = b/32
    with metadata (head dim D): k(b) = (2bD + 4D) / 64D, v(b) = (2bD + 64) / 64D
    uniform     HOT + WARM (k+v)
    hot_only    HOT
    fresh       HOT + WARM (k+v) + COLD k(c)          (refresh every step)
    stale       HOT + WARM (k+v) + COLD k(c) / R
    promote     HOT + (WARM - PW)(k+v) + PW (k(c) + 1) + COLD k(1) + PC + COLD k(c) / R
    twotier     HOT + (WARM - PW)(k+v) + PW (k(c) + 1)
where PW and PC are blocks promoted from WARM and COLD. The COLD k(c)/R term charges the
periodic refresh as a coarse key scan (the simulation refreshes from exact weights).

The go screen (docs/PHASE0.md, Decision) is applied to promote and twotier variants:
rho_codes <= 0.30, ppl_increase <= 1%, R >= 4 and effective HOT <= 20%.
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

GO = {"rho": 0.30, "ppl_increase": 0.01, "R": 4, "hot": 0.20}
TAIL = (0.02, 0.05, 0.10)


def costs(with_meta: bool, D: int):
    if not with_meta:
        return (lambda b: b / 32), (lambda b: b / 32)
    return (lambda b: (2 * b * D + 4 * D) / (64 * D)), (lambda b: (2 * b * D + 64) / (64 * D))


def rho(v: dict, fr: dict, with_meta: bool, D: int) -> float:
    k, val = costs(with_meta, D)
    kind, c, R, s = v["kind"], v["bits"], v["R"], v.get("scan_bits", 1)
    H, W, C, PW, PC = fr["hot"], fr["warm"], fr["cold"], fr["prom_warm"], fr["prom_cold"]
    if kind == "fp16":
        return 1.0
    if kind == "uniform":
        return H + W * (k(c) + val(c))
    if kind == "hot_only":
        return H
    if kind == "fresh":
        return H + W * (k(c) + val(c)) + C * k(c)
    if kind == "stale":
        return H + W * (k(c) + val(c)) + C * k(c) / R
    if kind == "promote":
        return H + (W - PW) * (k(c) + val(c)) + PW * (k(c) + 1) + C * k(s) + PC + C * k(c) / R
    if kind == "twotier":
        return H + (W - PW) * (k(c) + val(c)) + PW * (k(c) + 1)
    raise ValueError(kind)


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
        for line in (run / "done.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            pid = json.loads(line)["id"]
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
                        row["task_score"] = score(info["metric"], pred, info["answers"], info.get("all_classes"))
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
        row = {
            "variant": name, "kind": v["kind"], "R": v["R"] if v["kind"] in ("stale", "promote", "twotier") else "",
            "eps_w": v["eps_w"] if v["kind"] not in ("fp16", "uniform") else "",
            "eps_c": v["eps_c"] if v["kind"] in ("fresh", "stale", "promote") else "",
            "scan_codec": v.get("scan_codec") if v["kind"] == "promote" else "",
            "tokens": int(a["tokens"]),
            "ppl": math.exp(a["nll"] / a["tokens"]),
            "ppl_increase": math.exp(a["nll"] / a["tokens"]) / fp_ppl - 1,
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
        row["go_screen"] = bool(
            v["kind"] in ("promote", "twotier") and row["rho_codes"] <= GO["rho"] and row["ppl_increase"] <= GO["ppl_increase"]
            and v["R"] >= GO["R"] and hot_eff <= GO["hot"]
        )
        rows.append(row)
        layer_rows += [{"variant": name, "layer": i, "attn_err": float(e)} for i, e in enumerate(err_layers)]
    order = ["fp16", "uniform", "hot_only", "fresh", "stale", "promote", "twotier"]
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
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        commit = None
    (args.out / "meta.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "runs": [str(r) for r in args.runs], "prompts": n_prompts,
        "git_commit": commit, "go_criteria": GO, "fp16_ppl": fp_ppl, "run_meta": metas}, indent=2))

    print(f"{n_prompts} prompts, {int(fp['tokens'])} reference tokens per variant, fp16 ppl {fp_ppl:.4f} -> {args.out}")
    hdr = f"{'variant':38} {'dPPL%':>7} {'agree':>6} {'aErr':>6} {'rho':>6} {'rhoM':>6} {'HOTeff':>6} {'fixup':>6} {'recC':>5} {'prec':>5} {'tail>.05 b/a':>13}"
    print(hdr)
    for r in rows:
        rc = r["recall_cold"]
        print(f"{r['variant']:38} {100 * r['ppl_increase']:7.2f} {r['tf_agree']:6.3f} {r['attn_err_mean']:6.4f} {r['rho_codes']:6.3f} "
              f"{r['rho_meta']:6.3f} {r['hot_effective']:6.3f} {r['fixup_rate']:6.4f} {rc if rc == '' else format(rc, '.2f'):>5} "
              f"{r['promotion_precision'] if r['promotion_precision'] == '' else format(r['promotion_precision'], '.2f'):>5} "
              f"{r['frac_qheads_cold_mass_gt_0.05_before']:6.4f}/{r['frac_qheads_cold_mass_gt_0.05_after']:.4f}"
              + ("  GO-screen" if r["go_screen"] else ""))


if __name__ == "__main__":
    main()
