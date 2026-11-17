"""Compare two analyses of the same prompt set, e.g. a CUDA rerun against the MPS results.

Reads per_prompt.csv and summary.csv from both directories (written by analyze_d.py) and
prints, for the variants present in both:

- per prompt and variant: perplexity under A and B, and the paired increase over fp16 under
  A and B with the difference in percentage points;
- the summary rows side by side (perplexity increase with CI, rho_meta, HOT/WARM/COLD,
  worst and best prompt);
- the largest per-prompt disagreements.

Writes compare_per_prompt.csv and compare_summary.csv into --out. Analysis only: nothing is
recomputed from captures.

  python phase0/compare_runs.py --a runs/cluster/granite-replicate4k/analysis \
      --b results/prescreen-4k/granite-3.3-2b --labels cuda mps --out runs/cluster/granite-replicate4k/analysis
"""
import argparse
import csv
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", type=Path, required=True, help="analysis dir A (e.g. the CUDA rerun)")
    ap.add_argument("--b", type=Path, required=True, help="analysis dir B (e.g. results/prescreen-4k/<model>)")
    ap.add_argument("--labels", nargs=2, default=["a", "b"])
    ap.add_argument("--variants", nargs="*", help="default: every variant present in both")
    ap.add_argument("--out", type=Path)
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
    hdr = f"  {'variant':44} {'':5} {'dPPL% [95% CI]':>22} {'rhoM':>6} {'agree':>6} {'HOTeff':>6} {'WARM':>6} {'COLD':>6} {'worst%':>6} {'best%':>6} {'n>2%':>4} {'n':>3}"
    print(hdr)
    if only_a or only_b:
        print(f"  (prompt sets differ: summary rows are over different prompts; compare the per-prompt rows)")
    for v in variants:
        for lab, s in ((la, sa[v]), (lb, sb[v])):
            g = {c: num(s.get(c)) for c in SUMMARY_COLS}
            summ.append({"variant": v, "run": lab, **g})
            ci = f"{100 * g['ppl_increase']:6.2f} [{100 * g['ppl_increase_ci_lo']:5.2f},{100 * g['ppl_increase_ci_hi']:5.2f}]"
            print(f"  {v if lab == la else '':44} {lab:5} {ci:>22} {g['rho_meta']:6.3f} {g['tf_agree']:6.3f} "
                  f"{g['hot_effective']:6.3f} {g['warm']:6.3f} {g['cold']:6.3f} {100 * g['ppl_increase_prompt_max']:6.2f} "
                  f"{100 * g['ppl_increase_prompt_min']:6.2f} {g['prompts_above_2pct']:4.0f} {g['prompts']:3.0f}")

    finite = [r for r in rows if math.isfinite(r["increase_diff_pp"])]
    nonfinite = [r for r in rows if not math.isfinite(r["increase_diff_pp"])]
    print(f"\nlargest per-prompt disagreements in paired increase ({len(nonfinite)} non-finite):")
    for r in sorted(finite, key=lambda r: -abs(r["increase_diff_pp"]))[: args.top]:
        print(f"  {r['variant']:44} {r['prompt']:40} {r['increase_diff_pp']:+7.2f} pp  (ppl {100 * r['ppl_rel_diff']:+.2f}%)")
    for r in nonfinite:
        print(f"  NON-FINITE {r['variant']} {r['prompt']}")
    for v in variants:
        d = [abs(r["increase_diff_pp"]) for r in finite if r["variant"] == v]
        f = [abs(r["ppl_rel_diff"]) for r in rows if r["variant"] == v and math.isfinite(r["ppl_rel_diff"])]
        if d:
            print(f"  {v:44} median |diff| {sorted(d)[len(d) // 2]:.3f} pp, max {max(d):.3f} pp; "
                  f"median |ppl rel diff| {100 * sorted(f)[len(f) // 2]:.3f}%")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, rs in (("compare_per_prompt.csv", rows), ("compare_summary.csv", summ)):
            if rs:
                with open(args.out / name, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rs[0]))
                    w.writeheader()
                    w.writerows({k: (f"{x:.6g}" if isinstance(x, float) else x) for k, x in r.items()} for r in rs)
        print(f"\nwrote {args.out}/compare_per_prompt.csv and compare_summary.csv")


if __name__ == "__main__":
    main()
