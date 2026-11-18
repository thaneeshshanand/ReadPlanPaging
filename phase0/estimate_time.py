"""Estimate the wall time of the 7B uniform and ladder stages from a short timing run.

The timing stage of scripts/phase0_run.sh runs tiered.py on the first TIMING_N prompts of each
context length (fp16 + uniform4, then coldkeys_R4 + live on 4/3) into <out>/timing. This reads
those per-prompt timings (done.jsonl: prefill_s and per-variant decode seconds) and multiplies
them out over the prompt set, per stage and context length, counting only the work still to do
in <out> (prompts whose .npz for a variant already exists are skipped, as tiered.py does).

  python phase0/estimate_time.py --timing runs/gpu/x/timing --out-dir runs/gpu/x --prompts P.jsonl
  python phase0/estimate_time.py ... --gate uniform --max-hours 1   # exit 3 if the stage needs longer

It is an estimate: one or two prompts per length, first-prompt warm-up included, model loading
not counted. Writes <timing>/estimate.json.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Mirrors scripts/phase0_run.sh. Each pass is one tiered.py invocation: one prefill, then its variants.
UNIFORM = ["fp16", "uniform4", "uniform3", "uniformK4V3", "uniformK4V2", "uniformK3V4"]
LADDER_TIERED = ["coldkeys_R4", "live"]
STAGES = {
    "uniform": [{"variants": UNIFORM}],
    "ladder": [{"variants": LADDER_TIERED, "warm": w} for w in (3, 2)],
}
KIND_OF = {"fp16": "fp16", "coldkeys_R4": "coldkeys_R4", "live": "live"}  # every uniform* costs like uniform4


def timing_kind(v: str) -> str:
    return KIND_OF.get(v, "uniform4")


def load_timings(timing_dir: Path, n_tokens_of: dict):
    """Per context length: mean prefill seconds and mean decode seconds per timing kind."""
    pre, dec = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for line in (timing_dir / "done.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        L = n_tokens_of.get(r["id"], r["prompt_len"])
        pre[L].append(r["timings"]["prefill_s"])
        for k, s in r["timings"].items():
            if k != "prefill_s":
                base = k.split("_w")[0] if k.startswith(("coldkeys", "live")) else k.split("_k")[0]
                dec[L][timing_kind(base)].append(s)
    mean = lambda xs: sum(xs) / len(xs)
    return {L: {"prefill_s": mean(pre[L]), **{k: mean(v) for k, v in dec[L].items()}} for L in pre}


def variant_names(v: str, warm=None, kcodec="group256", eps_w="0.001", eps_c="0.02"):
    """The .npz name tiered.py writes for a variant (its Variant.name)."""
    if v == "fp16":
        return "fp16"
    if v in LADDER_TIERED:
        return f"{v}_w{eps_w}_c{eps_c}_h4b{warm}_k{kcodec}"
    return f"{v}_k{kcodec}"


def estimate(timings, records, out_dir: Path | None):
    est = {}
    for stage, passes in STAGES.items():
        per_len = {}
        for L in sorted({int(r["n_tokens"]) for r in records}):
            t = timings.get(L)
            recs = [r for r in records if int(r["n_tokens"]) == L]
            secs, todo_passes = 0.0, 0
            for r in recs:
                for ps in passes:
                    todo = [v for v in ps["variants"]
                            if not (out_dir and (out_dir / r["id"] / f"{variant_names(v, ps.get('warm'))}.npz").exists())]
                    if todo and t:
                        secs += t["prefill_s"] + sum(t.get(timing_kind(v), t.get("uniform4", 0.0)) for v in todo)
                    todo_passes += bool(todo)
            per_len[L] = {"prompts": len(recs), "passes_to_do": todo_passes,
                          "hours": secs / 3600 if t else None, "measured": t is not None}
        known = [x["hours"] for x in per_len.values() if x["hours"] is not None]
        est[stage] = {"by_length": per_len, "hours": sum(known),
                      "complete": all(x["measured"] or x["passes_to_do"] == 0 for x in per_len.values())}
    return est


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--timing", type=Path, required=True, help="the timing stage's tiered.py output dir")
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None, help="the stages' output dir; finished work is not counted")
    ap.add_argument("--gate", choices=list(STAGES), default=None)
    ap.add_argument("--max-hours", type=float, default=1.0)
    args = ap.parse_args()
    records = [json.loads(l) for l in args.prompts.read_text().splitlines() if l.strip()]
    if not (args.timing / "done.jsonl").exists():
        print(f"no timing results in {args.timing}: run the timing stage first", file=sys.stderr)
        return 3
    timings = load_timings(args.timing, {r["id"]: int(r["n_tokens"]) for r in records})
    est = estimate(timings, records, args.out_dir)
    print("measured per prompt (s): " + "; ".join(
        f"{L}: " + ", ".join(f"{k} {v:.1f}" for k, v in t.items()) for L, t in sorted(timings.items())))
    for stage, e in est.items():
        parts = [f"{L}: {x['passes_to_do']} passes, " + (f"{x['hours']:.2f} h" if x["hours"] is not None else "not measured")
                 for L, x in e["by_length"].items()]
        print(f"ESTIMATE {stage:8} {e['hours']:6.2f} h remaining  ({'; '.join(parts)})")
    (args.timing / "estimate.json").write_text(json.dumps(
        {"timings_s": {str(k): v for k, v in timings.items()}, "stages": est,
         "note": "estimate from the timing stage; model loading not counted"}, indent=2, default=str))
    if args.gate:
        e = est[args.gate]
        if not e["complete"]:
            print(f"GATE {args.gate}: some context lengths have no timing; run the timing stage first", file=sys.stderr)
            return 3
        if e["hours"] > args.max_hours:
            print(f"GATE {args.gate}: estimated {e['hours']:.2f} h > MAX_HOURS={args.max_hours:g}. "
                  f"Confirm, then rerun with MAX_HOURS={e['hours'] * 1.3:.0f} or more.", file=sys.stderr)
            return 3
        print(f"GATE {args.gate}: {e['hours']:.2f} h <= MAX_HOURS={args.max_hours:g}, starting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
