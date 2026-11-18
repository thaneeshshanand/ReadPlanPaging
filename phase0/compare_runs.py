"""Compare two analyses of the same prompt set, e.g. a CUDA rerun against the MPS results.

Reads per_prompt.csv and summary.csv from both directories (written by analyze_d.py) and
prints, for the variants present in both:

- per prompt and variant: perplexity under A and B, and the paired increase over fp16 under
  A and B with the difference in percentage points;
- the summary rows side by side (perplexity increase with CI, rho_meta, HOT/WARM/COLD,
  worst and best prompt);
- the per-prompt drift as a distribution per variant (quantiles of |difference| and counts
  by band), and the largest disagreements;
- the replication check (--check): PASS if the verdict in both decision.json files is the
  same, every pooled perplexity change is within --pooled-pp points (default 0.25) of B's,
  and no prompt's paired increase differs by more than --prompt-pp points (default 5) or is
  non-finite. Exit code 2 on FAIL. Per-prompt drift below --prompt-pp is reported, never flagged:
  it is ~1 point between torch versions (results/prescreen-4k/granite-3.3-2b-torch2.15/NOTES.md).

Writes compare_per_prompt.csv, compare_summary.csv, compare_drift.csv and (with --check)
compare_check.json into --out. Analysis only: nothing is
recomputed from captures.

  python phase0/compare_runs.py --check --a runs/gpu/granite-3.3-2b-replicate4k/analysis \
      --b results/prescreen-4k/granite-3.3-2b --labels cuda mps --out runs/gpu/granite-3.3-2b-replicate4k/analysis
"""
import argparse
import csv
import json
import math
from pathlib import Path

SUMMARY_COLS = ("ppl", "ppl_increase", "ppl_increase_ci_lo", "ppl_increase_ci_hi", "rho_meta", "tf_agree",
                "hot_effective", "warm", "cold", "ppl_increase_prompt_max", "ppl_increase_prompt_min", "prompts_above_2pct", "prompts")


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load(d: Path):
    pp = {(r["variant"], r["prompt"]): r for r in csv.DictReader(open(d / "per_prompt.csv"))}
    summ = {r["variant"]: r for r in csv.DictReader(open(d / "summary.csv"))}
    return pp, summ


def ppl(r):
    t = num(r["tokens"])
    return math.exp(num(r["nll_sum"]) / t) if t else float("nan")


BANDS = ((0, 0.3), (0.3, 1), (1, 2), (2, 5), (5, math.inf))
BAND_NAMES = ("<0.3", "0.3-1", "1-2", "2-5", ">5")


def quantile(xs, q):
    """Linear-interpolated quantile of a non-empty list."""
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = math.floor(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


def drift_table(rows, variants):
    """Per variant: quantiles of |per-prompt paired increase difference| (points) and counts per band."""
    out = []
    for v in variants:
        d = [abs(r["increase_diff_pp"]) for r in rows if r["variant"] == v and math.isfinite(r["increase_diff_pp"])]
        if not d:
            continue
        row = {"variant": v, "n": len(d), "median": quantile(d, 0.5), "p75": quantile(d, 0.75),
               "p90": quantile(d, 0.9), "max": max(d)}
        row.update({name: sum(lo <= x < hi for x in d) for name, (lo, hi) in zip(BAND_NAMES, BANDS)})
        out.append(row)
    return out


def replication_check(dir_a, dir_b, sa, sb, variants, rows, pooled_pp, prompt_pp, only_a=(), only_b=()):
    """Same verdict, pooled increases within pooled_pp points, no prompt beyond prompt_pp points."""
    verdict = {}
    for k, d in (("a", dir_a), ("b", dir_b)):
        f = Path(d) / "decision.json"
        verdict[k] = json.loads(f.read_text())["verdict"] if f.exists() else None
    pooled = {v: 100 * (num(sa[v]["ppl_increase"]) - num(sb[v]["ppl_increase"])) for v in variants}
    worst_v = max(pooled, key=lambda v: abs(pooled[v]) if math.isfinite(pooled[v]) else math.inf) if pooled else None
    diffs = [r["increase_diff_pp"] for r in rows]
    bad = [r for r in rows if not math.isfinite(r["increase_diff_pp"]) or abs(r["increase_diff_pp"]) > prompt_pp]
    worst_r = max(rows, key=lambda r: abs(r["increase_diff_pp"]) if math.isfinite(r["increase_diff_pp"]) else math.inf, default=None)
    crit = [
        {"name": "verdict unchanged", "ok": verdict["a"] is not None and verdict["a"] == verdict["b"],
         "detail": f"{verdict['a']!r} vs {verdict['b']!r}"},
        {"name": f"pooled ppl increase within {pooled_pp} points",
         "ok": bool(pooled) and all(math.isfinite(x) and abs(x) <= pooled_pp for x in pooled.values()),
         "detail": (f"largest {pooled[worst_v]:+.3f} pp ({worst_v})" if worst_v else "no variants")
                   + "; " + ", ".join(f"{v} {x:+.3f}" for v, x in pooled.items())},
        {"name": f"no prompt beyond {prompt_pp} points",
         "ok": bool(rows) and not bad,
         "detail": (f"{len(bad)} of {len(rows)} rows beyond or non-finite; largest {worst_r['increase_diff_pp']:+.2f} pp "
                    f"({worst_r['variant']}, {worst_r['prompt']})" if worst_r else "no paired rows")},
        {"name": "same prompt set", "ok": not only_a and not only_b,
         "detail": f"{len(only_a)} prompts only in A, {len(only_b)} only in B"},
    ]
    return {"pass": all(c["ok"] for c in crit), "criteria": crit, "pooled_diff_pp": pooled,
            "rows_beyond": [{"variant": r["variant"], "prompt": r["prompt"], "diff_pp": r["increase_diff_pp"]} for r in bad],
            "thresholds": {"pooled_pp": pooled_pp, "prompt_pp": prompt_pp}, "a": str(dir_a), "b": str(dir_b)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", type=Path, required=True, help="analysis dir A (e.g. the CUDA rerun)")
    ap.add_argument("--b", type=Path, required=True, help="analysis dir B (e.g. results/prescreen-4k/<model>)")
    ap.add_argument("--labels", nargs=2, default=["a", "b"])
    ap.add_argument("--variants", nargs="*", help="default: every variant present in both")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check", action="store_true", help="apply the replication pass rule; exit 2 on FAIL")
    ap.add_argument("--pooled-pp", type=float, default=0.25, help="pass: max |pooled ppl increase difference|, points")
    ap.add_argument("--prompt-pp", type=float, default=5.0, help="pass: max |per-prompt paired increase difference|, points")
    ap.add_argument("--top", type=int, default=10, help="largest per-prompt disagreements to print")
    args = ap.parse_args()
    la, lb = args.labels
    ppa, sa = load(args.a)
    ppb, sb = load(args.b)
    variants = args.variants or [v for v in sa if v in sb]
    missing = [v for v in (args.variants or []) if v not in sa or v not in sb]
    if missing:
        print(f"not in both: {missing}")
        variants = [v for v in variants if v not in missing]

    rows = []
    for (v, p), ra in sorted(ppa.items()):
        if v not in variants or (v, p) not in ppb:
            continue
        rb = ppb[(v, p)]
        ia, ib = num(ra["ppl_increase"]), num(rb["ppl_increase"])
        rows.append({"variant": v, "prompt": p, f"ppl_{la}": ppl(ra), f"ppl_{lb}": ppl(rb),
                     f"ppl_rel_diff": ppl(ra) / ppl(rb) - 1,
                     f"increase_{la}": ia, f"increase_{lb}": ib, "increase_diff_pp": 100 * (ia - ib),
                     f"tf_agree_{la}": num(ra["tf_agree"]), f"tf_agree_{lb}": num(rb["tf_agree"])})
    only_a = sorted({p for (v, p) in ppa if v in variants} - {p for (v, p) in ppb if v in variants})
    only_b = sorted({p for (v, p) in ppb if v in variants} - {p for (v, p) in ppa if v in variants})

    print(f"A = {la}: {args.a}\nB = {lb}: {args.b}")
    print(f"{len(rows)} paired (variant, prompt) rows; prompts only in {la}: {len(only_a)}, only in {lb}: {len(only_b)}")

    print(f"\nper prompt: perplexity under {la} / {lb} and paired increase over fp16 (percent)")
    prompts = sorted({r["prompt"] for r in rows})
    by = {(r["variant"], r["prompt"]): r for r in rows}
    for v in variants:
        print(f"\n  {v}")
        print(f"    {'prompt':40} {'ppl ' + la:>10} {'ppl ' + lb:>10} {'rel':>7} {'inc ' + la:>8} {'inc ' + lb:>8} {'diff pp':>8}")
        for p in prompts:
            r = by.get((v, p))
            if r:
                print(f"    {p:40} {r[f'ppl_{la}']:10.4f} {r[f'ppl_{lb}']:10.4f} {100 * r['ppl_rel_diff']:6.2f}% "
                      f"{100 * r[f'increase_{la}']:8.2f} {100 * r[f'increase_{lb}']:8.2f} {r['increase_diff_pp']:8.2f}")

    print("\nsummary rows side by side")
    summ = []
    hdr = f"  {'variant':44} {'':5} {'dPPL% [95% CI]':>22} {'rhoM':>6} {'agree':>6} {'HOTeff':>6} {'WARM':>6} {'COLD':>6} {'worst%†':>7} {'best%†':>6} {'n>2%†':>5} {'n':>3}"
    print(hdr)
    if only_a or only_b:
        print(f"  (prompt sets differ: summary rows are over different prompts; compare the per-prompt rows)")
    for v in variants:
        for lab, s in ((la, sa[v]), (lb, sb[v])):
            g = {c: num(s.get(c)) for c in SUMMARY_COLS}
            summ.append({"variant": v, "run": lab, **g})
            ci = f"{100 * g['ppl_increase']:6.2f} [{100 * g['ppl_increase_ci_lo']:5.2f},{100 * g['ppl_increase_ci_hi']:5.2f}]"
            print(f"  {v if lab == la else '':44} {lab:5} {ci:>22} {g['rho_meta']:6.3f} {g['tf_agree']:6.3f} "
                  f"{g['hot_effective']:6.3f} {g['warm']:6.3f} {g['cold']:6.3f} {100 * g['ppl_increase_prompt_max']:7.2f} "
                  f"{100 * g['ppl_increase_prompt_min']:6.2f} {g['prompts_above_2pct']:5.0f} {g['prompts']:3.0f}")
    print("  † indicative only: per-prompt figures (worst, best, prompts above +2%) are noisy and do not decide")

    finite = [r for r in rows if math.isfinite(r["increase_diff_pp"])]
    nonfinite = [r for r in rows if not math.isfinite(r["increase_diff_pp"])]
    drift = drift_table(rows, variants)
    print(f"\nper-prompt drift in paired increase, |{la} - {lb}| in points ({len(nonfinite)} non-finite rows)")
    print(f"  {'variant':44} {'n':>3} {'median':>6} {'p75':>6} {'p90':>6} {'max':>6}  " + " ".join(f"{b:>7}" for b in BAND_NAMES))
    for d in drift:
        print(f"  {d['variant']:44} {d['n']:3d} {d['median']:6.2f} {d['p75']:6.2f} {d['p90']:6.2f} {d['max']:6.2f}  "
              + " ".join(f"{d[b]:7d}" for b in BAND_NAMES))
    print(f"\nlargest per-prompt disagreements:")
    for r in sorted(finite, key=lambda r: -abs(r["increase_diff_pp"]))[: args.top]:
        print(f"  {r['variant']:44} {r['prompt']:40} {100 * r[f'increase_{la}']:7.2f} vs {100 * r[f'increase_{lb}']:7.2f}"
              f"  ({r['increase_diff_pp']:+.2f} pp)")
    for r in nonfinite:
        print(f"  NON-FINITE {r['variant']} {r['prompt']}")

    check = None
    if args.check:
        check = replication_check(args.a, args.b, sa, sb, variants, rows, args.pooled_pp, args.prompt_pp, only_a, only_b)
        print(f"\nreplication check: {'PASS' if check['pass'] else 'FAIL'}")
        for c in check["criteria"]:
            print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, rs in (("compare_per_prompt.csv", rows), ("compare_summary.csv", summ), ("compare_drift.csv", drift)):
            if rs:
                with open(args.out / name, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rs[0]))
                    w.writeheader()
                    w.writerows({k: (f"{x:.6g}" if isinstance(x, float) else x) for k, x in r.items()} for r in rs)
        if check:
            (args.out / "compare_check.json").write_text(json.dumps(check, indent=2))
        print(f"\nwrote compare_*.csv{' and compare_check.json' if check else ''} to {args.out}")
    return 2 if check and not check["pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
