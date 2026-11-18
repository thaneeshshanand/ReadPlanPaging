#!/usr/bin/env python
"""Analysis for docs/JUMPS.md: memory, random and lexical recovery of the jumps variant e's stale plan misses.

Reads jumps_capture.py output (one directory per model, plus optional positive-control
directories) and writes, per model, per_head.csv, per_layer.csv and summary.json, and overall
decision.json, under --out. Definitions, controls and the rule are in docs/JUMPS.md and are
not parameters here.

  python phase0/jumps_analyze.py --runs runs/jumps/qwen2.5-1.5b runs/jumps/qwen2.5-3b runs/jumps/granite-3.3-2b \
      --controls runs/jumps/qwen2.5-1.5b-repeat runs/jumps/qwen2.5-3b-repeat runs/jumps/granite-3.3-2b-repeat \
      --out runs/jumps/analysis
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

COLD = 0  # analyze.py tier codes: COLD 0, WARM 1, HOT 2
BLOCK = 16
CAP_FRAC = 0.05
MEMORY_WINDOW = 32
MAX_SUFFIX = 4
TOP_FRACS = (0.05, 0.10, 0.20)
FIRE_SHARE = 0.80  # positive control, original definition (reported): heads holding this share of leftover jump mass
SHARP = 0.50  # positive control (amendment 2): a head fires if its largest block holds >= this share of jump mass
POS_PASS = 0.90
NEG_BAND = (0.5, 2.0)
RULE_CONC, RULE_LEX, RULE_RATIO = 0.90, 0.40, 5.0

FIELDS = ["prompt", "t", "layer", "kvh", "qh", "jump", "mem", "left", "rand", "lex_exact", "lex_norm", "lex_neg", "top1",
          "rand_m_exact", "rand_m_norm", "rand_m_neg",  # rand_m_*: random with as many blocks as that lookup picked
          "ceil", "first4", "behind8"]  # addition 1 (docs/JUMPS.md): ceiling and position; reported, do not decide
REFRESH_R = 4
FIRST_BLOCKS, BEHIND_BLOCKS = 4, 8


def cap_of(nb: int) -> int:
    return int(CAP_FRAC * nb)


# --------------------------------------------------------------------------- selection rules


def memory_blocks(plan_hist: np.ndarray, cold_now: np.ndarray, cap: int) -> np.ndarray:
    """plan_hist [S, nb] plans of the previous S <= 32 steps, oldest first (PAD = not yet created,
    counts as not COLD); cold_now [nb] bool. Returns block indices, most recent non-COLD first."""
    if cap <= 0 or len(plan_hist) == 0:
        return np.zeros(0, np.int64)
    noncold = plan_hist != COLD
    seen = noncold.any(0) & cold_now
    if not seen.any():
        return np.zeros(0, np.int64)
    S = len(plan_hist)
    last = np.where(noncold, np.arange(S)[:, None], -1).max(0)
    b = np.nonzero(seen)[0]
    order = np.lexsort((-b, -last[b]))  # primary: most recent step, then newest block
    return b[order][:cap]


def lexical_blocks(query: np.ndarray, pos: int, index: np.ndarray, logcount: dict) -> list[int]:
    """Ranked, de-duplicated blocks for the token at query[pos].

    index: the key sequence searched (the same sequence, or another prompt's for the negative
    control), positions < pos only. Suffix match compares index[p - j] with query[pos - j]."""
    x = query[pos]
    hay = index[:pos]
    P = np.nonzero(hay == x)[0]
    if len(P) == 0:
        return []
    k = np.ones(len(P), np.int64)
    alive = np.ones(len(P), bool)
    for j in range(1, MAX_SUFFIX):
        ok = alive & (P - j >= 0) & (pos - j >= 0)
        ok[ok] = index[P[ok] - j] == query[pos - j]
        k += ok
        alive = ok
    rarity = np.zeros(len(P))
    for j in range(MAX_SUFFIX):
        m = k > j
        rarity[m] -= np.array([logcount.get(int(v), 0.0) for v in index[P[m] - j]])
    order = np.lexsort((-P, -rarity, -k))
    out, seen = [], set()
    for p in P[order]:
        for b in (p // BLOCK, (p + 1) // BLOCK):
            if b not in seen:
                seen.add(b)
                out.append(int(b))
    return out


def norm_keys(tok, ids: np.ndarray) -> np.ndarray:
    """Token text, lowercased, leading spaces stripped, as integer keys. Partial UTF-8 byte tokens
    (decode to U+FFFD) keep their raw token string so they stay distinct."""
    table = {}
    keys = np.empty(len(ids), np.int64)
    uniq = np.unique(ids)
    for i in uniq:
        s = tok.decode([int(i)])
        s = tok.convert_ids_to_tokens(int(i)) if "�" in s else s.lower().lstrip(" ")
        table[int(i)] = s
    vocab = {s: n for n, s in enumerate(sorted(set(table.values())))}
    for n, i in enumerate(ids):
        keys[n] = vocab[table[int(i)]]
    return keys


def log_counts(keys: np.ndarray) -> dict:
    u, c = np.unique(keys, return_counts=True)
    return {int(a): math.log(b) for a, b in zip(u, c)}


# --------------------------------------------------------------------------- per prompt


def load_run(d: Path):
    meta = json.loads((d / "meta.json").read_text())
    ids = [json.loads(l)["id"] for l in (d / "done.jsonl").read_text().splitlines() if l.strip()]
    return meta, ids


def analyse_prompt(pi: int, pdir: Path, tok, other: Path | None):
    z = np.load(pdir / "jumps.npz")
    info = json.loads((pdir / "prompt.json").read_text())
    n0 = info["prompt_len"]
    ctx = np.asarray(info["context_ids"], np.int64)
    plan, jidx, jmass = z["plan"], z["jump_idx"], z["jump_mass"].astype(np.float32)
    T, L, H, _ = plan.shape
    Hq = z["cold"].shape[2]
    G = Hq // H
    modes = {"exact": ctx}
    if tok is not None:
        modes["norm"] = norm_keys(tok, ctx)
    lc = {m: log_counts(k[:n0]) for m, k in modes.items()}
    neg = None
    if other is not None:
        octx = np.asarray(json.loads((other / "prompt.json").read_text())["context_ids"], np.int64)
        neg = (octx, log_counts(octx[: json.loads((other / "prompt.json").read_text())["prompt_len"]]))
    rows, lookup = [], []
    groups = {}
    for r, (t, l, q) in enumerate(jidx):
        groups.setdefault((int(t), int(l), int(q) // G), []).append(r)
    cand_cache = {}
    for (t, l, h), rs in sorted(groups.items()):
        pos = n0 + t  # position of the input token at step t
        nb = math.ceil((pos + 1) / BLOCK)
        cap = cap_of(nb)
        row = plan[t, l, h, :nb]
        cold = row == COLD
        mem = memory_blocks(plan[max(0, t - MEMORY_WINDOW):t, l, h, :nb], cold, cap)
        rem = cold.copy()
        rem[mem] = False
        nrem = int(rem.sum())
        k2 = min(cap, nrem)
        if t not in cand_cache:
            c = {m: lexical_blocks(k, pos, k, lc[m]) for m, k in modes.items()}
            if neg is not None:
                n = min(len(neg[0]), pos)
                c["neg"] = [b for b in lexical_blocks(ctx, pos, neg[0][:n], neg[1]) if b < nb]
            cand_cache[t] = c
            lookup.append({"prompt": pi, "t": t, **{f"returned_{m}": len(v) for m, v in c.items()}})
        sel = {}
        for m, c in cand_cache[t].items():
            inrem = [b for b in c if rem[b]]
            sel[m] = np.asarray(inrem[:k2], np.int64)
            lookup.append({"prompt": pi, "t": t, "layer": l, "kvh": h, f"in_cold_{m}": len(inrem)})
        nan = float("nan")
        nb_ref = math.ceil((n0 + t - t % REFRESH_R + 1) / BLOCK)  # blocks at the plan's refresh step
        recent = nb_ref - 2  # the last two blocks at refresh and every newer one are HOT (always read)
        behind = slice(max(0, recent - BEHIND_BLOCKS), max(0, recent))
        remidx = np.nonzero(rem)[0]
        for r in rs:
            m = jmass[r, :nb]
            J = float(m.sum())
            mc = float(m[mem].sum())
            left = J - mc
            rows.append([pi, t, l, h, int(jidx[r, 2]), J, mc, left, left * k2 / nrem if nrem else 0.0,
                         float(m[sel["exact"]].sum()),
                         float(m[sel["norm"]].sum()) if "norm" in sel else float("nan"),
                         float(m[sel["neg"]].sum()) if "neg" in sel else nan, float(m.max())]
                        + [left * len(sel[k]) / nrem if k in sel and nrem else (0.0 if k in sel else nan)
                           for k in ("exact", "norm", "neg")]
                        + [float(np.sort(m[remidx])[::-1][:k2].sum()), float(m[:FIRST_BLOCKS].sum()), float(m[behind].sum())])
    return np.asarray(rows, np.float64).reshape(-1, len(FIELDS)), lookup, (L, H)


# --------------------------------------------------------------------------- aggregation


def col(a, name):
    return a[:, FIELDS.index(name)]


def shares(a) -> dict:
    left = col(a, "left").sum()
    out = {"jumps": int(len(a)), "jump_mass": float(col(a, "jump").sum()), "mem_covered": float(col(a, "mem").sum()),
           "leftover": float(left)}
    for k in ("rand", "lex_exact", "lex_norm", "lex_neg", "rand_m_exact", "rand_m_norm", "rand_m_neg"):
        v = col(a, k)
        out[f"{k}_share"] = float(np.nansum(v) / left) if left > 0 and not np.isnan(v).all() else None
    for k in ("lex_exact", "lex_norm"):
        out[f"{k}_over_rand"] = (out[f"{k}_share"] / out["rand_share"]
                                 if out[f"{k}_share"] is not None and out["rand_share"] else None)
    for k in ("exact", "norm", "neg"):  # against random with the same number of blocks as the lookup picked
        a_, b_ = out[f"lex_{k}_share"], out[f"rand_m_{k}_share"]
        out[f"lex_{k}_over_rand_matched"] = a_ / b_ if a_ is not None and b_ else None
    out["mem_share_of_jump"] = float(col(a, "mem").sum() / col(a, "jump").sum()) if len(a) else None
    # addition 1: reported beside the rule, do not decide
    J = col(a, "jump")
    out["ceiling_share"] = float(col(a, "ceil").sum() / left) if left > 0 else None
    sharp = col(a, "top1") >= SHARP * J
    out["sharp_share_of_jumps"] = float(sharp.mean()) if len(a) else None
    out["sharp_share_of_jump_mass"] = float(J[sharp].sum() / J.sum()) if len(a) else None
    out["first4_share_of_jump_mass"] = float(col(a, "first4").sum() / J.sum()) if len(a) else None
    out["behind8_share_of_jump_mass"] = float(col(a, "behind8").sum() / J.sum()) if len(a) else None
    return out


def head_mass(a, L, H) -> np.ndarray:
    m = np.zeros(L * H)
    np.add.at(m, (col(a, "layer") * H + col(a, "kvh")).astype(int), col(a, "jump"))
    return m


def topk(frac, n):
    return max(1, round(frac * n))


def table(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fmt(x, p=1):
    return "-" if x is None else f"{100 * x:.{p}f}%"


def analyse_run(d: Path, tok_loader, neg: bool):
    meta, ids = load_run(d)
    tok = tok_loader(meta["model"], meta.get("model_revision"))
    allrows, lookups = [], []
    for i, pid in enumerate(ids):
        other = d / ids[(i + 1) % len(ids)] if neg and len(ids) > 1 else None
        rows, lk, (L, H) = analyse_prompt(i, d / pid, tok, other)
        allrows.append(rows)
        lookups += lk
    return meta, ids, np.concatenate(allrows), lookups, (L, H)


def lookup_stats(lookups, key):
    v = np.asarray([x[key] for x in lookups if key in x], float)
    if not len(v):
        return None
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)),
            "max": float(v.max()), "zero_share": float((v == 0).mean()), "n": int(len(v))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=Path, nargs="+", required=True, help="jumps_capture.py dirs, one per model")
    ap.add_argument("--controls", type=Path, nargs="*", default=[], help="positive-control (--repeat) dirs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-norm", action="store_true", help="skip the normalised-text lookup (no tokenizer needed)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    toks = {}

    def tok_loader(name, rev):
        if args.no_norm:
            return None
        if name not in toks:
            from transformers import AutoTokenizer
            toks[name] = AutoTokenizer.from_pretrained(name, revision=rev)
        return toks[name]

    doc = Path(__file__).resolve().parent.parent / "docs" / "JUMPS.md"
    decision = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "prereg_sha256": hashlib.sha256(doc.read_bytes()).hexdigest(),
                "controls": {}, "models": {}}

    # ---- positive controls
    for d in args.controls:
        meta, ids, a, lk, (L, H) = analyse_run(d, tok_loader, neg=False)
        left = np.zeros(L * H)
        np.add.at(left, (col(a, "layer") * H + col(a, "kvh")).astype(int), col(a, "left"))
        order = np.argsort(-left)
        csum = np.cumsum(left[order]) / max(left.sum(), 1e-30)
        top80 = order[: int(np.searchsorted(csum, FIRE_SHARE) + 1)]
        hid = (col(a, "layer") * H + col(a, "kvh")).astype(int)
        jm, t1 = np.zeros(L * H), np.zeros(L * H)
        np.add.at(jm, hid, col(a, "jump"))
        np.add.at(t1, hid, col(a, "top1"))
        fire = np.nonzero((jm > 0) & (t1 >= SHARP * np.maximum(jm, 1e-30)))[0]
        f = np.isin(hid, fire)
        s_all, s_fire, s_80 = shares(a), shares(a[f]), shares(a[np.isin(hid, top80)])
        res = {"model": meta["model"], "dir": str(d), "prompts": len(ids), "all_heads": s_all,
               "firing_heads": [[int(x // H), int(x % H)] for x in fire], "fire": s_fire,
               "fire_share_of_jump_mass": float(jm[fire].sum() / jm.sum()),
               "original_definition_top80": {"heads": len(top80), **s_80},
               "pass": bool(len(fire) and s_fire["lex_exact_share"] is not None and s_fire["lex_exact_share"] >= POS_PASS),
               "pass_original_definition": bool(s_80["lex_exact_share"] is not None and s_80["lex_exact_share"] >= POS_PASS)}
        decision["controls"][meta["model"]] = {"positive": res}
        print(f"POSITIVE {meta['model']}: {len(ids)} prompts, {s_all['jumps']} jumps, firing heads {len(fire)}/{L * H}; "
              f"exact lexical covers {fmt(s_fire['lex_exact_share'])} of leftover in them "
              f"(they hold {fmt(res['fire_share_of_jump_mass'])} of jump mass; random {fmt(s_fire['rand_share'])}; all heads "
              f"{fmt(s_all['lex_exact_share'])}; original top-80% definition {fmt(s_80['lex_exact_share'])}) -> "
              f"{'PASS' if res['pass'] else 'FAIL'}")

    # ---- models
    verdict_inputs = []
    for d in args.runs:
        meta, ids, a, lk, (L, H) = analyse_run(d, tok_loader, neg=True)
        name = meta["model"]
        mdir = args.out / name.split("/")[-1].lower()
        mdir.mkdir(parents=True, exist_ok=True)
        np.save(mdir / "jumps_rows.npy", a)
        nH = L * H
        # per head and layer, all prompts (do not decide)
        hr, lr = [], []
        for l in range(L):
            al = a[col(a, "layer") == l]
            s = shares(al)
            lr.append({"layer": l, **{k: s[k] for k in ("jumps", "jump_mass", "leftover", "rand_share", "lex_exact_share",
                                                        "lex_norm_share", "lex_neg_share", "ceiling_share",
                                                        "sharp_share_of_jump_mass", "first4_share_of_jump_mass",
                                                        "behind8_share_of_jump_mass")}})
            for h in range(H):
                ah = al[col(al, "kvh") == h]
                s = shares(ah)
                hr.append({"layer": l, "kvh": h, **{k: s[k] for k in ("jumps", "jump_mass", "mem_covered", "leftover",
                                                                       "rand_share", "lex_exact_share", "lex_norm_share",
                                                                       "lex_neg_share", "ceiling_share",
                                                                       "sharp_share_of_jump_mass",
                                                                       "first4_share_of_jump_mass",
                                                                       "behind8_share_of_jump_mass")}})
        table(mdir / "per_head.csv", hr)
        table(mdir / "per_layer.csv", lr)
        # negative control, all prompts
        s_all = shares(a)
        nratio = s_all["lex_neg_over_rand_matched"]
        neg_pass = nratio is not None and NEG_BAND[0] <= nratio <= NEG_BAND[1]
        # held-out split
        pa = col(a, "prompt").astype(int)
        A, B = a[pa % 2 == 0], a[pa % 2 == 1]
        mA, mB = head_mass(A, L, H), head_mass(B, L, H)
        rank = np.argsort(-mA)
        conc = {f"top{int(f * 100)}pct": {"heads": topk(f, nH), "share_B": float(mB[rank[:topk(f, nH)]].sum() / mB.sum())}
                for f in TOP_FRACS}
        top10 = rank[: topk(0.10, nH)]
        hB = (col(B, "layer") * H + col(B, "kvh")).astype(int)
        outside = B[~np.isin(hB, top10)]
        s_out = shares(outside)
        scan_left = float(col(outside, "left").sum() / col(B, "jump").sum())
        noscan_left = float(col(B, "left").sum() / col(B, "jump").sum())
        lex_ok = (s_out["lex_exact_share"] is not None and s_out["lex_exact_share"] >= RULE_LEX
                  and s_out["lex_exact_over_rand"] is not None and s_out["lex_exact_over_rand"] >= RULE_RATIO)
        lex_ok_norm = (s_out["lex_norm_share"] is not None and s_out["lex_norm_share"] >= RULE_LEX
                       and s_out["lex_norm_over_rand"] is not None and s_out["lex_norm_over_rand"] >= RULE_RATIO)
        summ = {
            "model": name, "revision": meta.get("model_revision"), "dir": str(d), "prompts": len(ids),
            "kv_heads_total": nH, "all_prompts": s_all,
            "negative_control": {"lex_neg_share": s_all["lex_neg_share"], "rand_matched_share": s_all["rand_m_neg_share"],
                                 "rand_full_cap_share": s_all["rand_share"], "ratio_to_matched": nratio,
                                 "band": NEG_BAND, "pass": bool(neg_pass)},
            "lookup_returned_per_step": {m: lookup_stats(lk, f"returned_{m}") for m in ("exact", "norm", "neg")},
            "lookup_in_remaining_cold_per_head_step": {m: lookup_stats(lk, f"in_cold_{m}") for m in ("exact", "norm", "neg")},
            "cap_blocks_at_end": cap_of(math.ceil((json.loads((d / ids[0] / "prompt.json").read_text())["prompt_len"] + 128) / BLOCK)),
            "concentration_heldout_B": conc,
            "top10_heads_from_A": [[int(x // H), int(x % H)] for x in top10],
            "half_B": shares(B), "half_B_outside_top10": s_out,
            "scan_top10_leftover_share_B": scan_left, "plan_memory_leftover_share_B": noscan_left,
            "rule": {"concentrated": conc["top10pct"]["share_B"] >= RULE_CONC, "lexical_pass_exact": bool(lex_ok),
                     "lexical_pass_norm": bool(lex_ok_norm)},
        }
        (mdir / "summary.json").write_text(json.dumps(summ, indent=2))
        decision["models"][name] = summ
        verdict_inputs.append(summ["rule"])
        ls = summ["lookup_returned_per_step"]["exact"]
        print(f"\n== {name}: {len(ids)} prompts, {s_all['jumps']} jumps, jump mass {s_all['jump_mass']:.1f}, "
              f"memory covers {fmt(s_all['mem_share_of_jump'])}")
        print(f"  all prompts: leftover covered by random {fmt(s_all['rand_share'])}, lexical exact "
              f"{fmt(s_all['lex_exact_share'])}, norm {fmt(s_all['lex_norm_share'])}")
        print(f"  negative control: lexical from another prompt {fmt(s_all['lex_neg_share'])} vs random with as many "
              f"blocks {fmt(s_all['rand_m_neg_share'])} (x{nratio or float('nan'):.2f}) -> {'PASS' if neg_pass else 'FAIL'}")
        print(f"  lexical exact vs random with as many blocks: {fmt(s_all['lex_exact_share'])} vs "
              f"{fmt(s_all['rand_m_exact_share'])} (x{(s_all['lex_exact_over_rand_matched'] or float('nan')):.1f})")
        print(f"  lookup returns per step (exact): mean {ls['mean']:.1f}, median {ls['median']:.0f}, p90 {ls['p90']:.0f}, "
              f"none {fmt(ls['zero_share'])}; cap {summ['cap_blocks_at_end']}")
        print("  held-out half B, top heads by half-A rank: " + ", ".join(
            f"{k} ({v['heads']}) {fmt(v['share_B'])}" for k, v in conc.items()))
        print(f"  half B outside top 10%: random {fmt(s_out['rand_share'])}, lexical exact {fmt(s_out['lex_exact_share'])} "
              f"(x{(s_out['lex_exact_over_rand'] or 0):.1f}), norm {fmt(s_out['lex_norm_share'])}")
        print(f"  leftover jump mass on B: plan+memory {fmt(noscan_left)}, scan top 10% + plan+memory {fmt(scan_left)}")
        print(f"  beside the rule (half B outside top 10%): ceiling {fmt(s_out['ceiling_share'])}; sharp jumps "
              f"{fmt(s_out['sharp_share_of_jumps'])} of jumps, {fmt(s_out['sharp_share_of_jump_mass'])} of mass; "
              f"first 4 blocks {fmt(s_out['first4_share_of_jump_mass'])}, 8 behind recent "
              f"{fmt(s_out['behind8_share_of_jump_mass'])}")
        print(f"  same, all prompts all heads: ceiling {fmt(s_all['ceiling_share'])}; sharp {fmt(s_all['sharp_share_of_jumps'])}"
              f" / {fmt(s_all['sharp_share_of_jump_mass'])}; first 4 {fmt(s_all['first4_share_of_jump_mass'])}; "
              f"behind 8 {fmt(s_all['behind8_share_of_jump_mass'])}")

    # ---- rule (docs/JUMPS.md)
    n_conc = sum(r["concentrated"] for r in verdict_inputs)
    n_lex = sum(r["lexical_pass_exact"] for r in verdict_inputs)
    n_lex_norm = sum(r["lexical_pass_norm"] for r in verdict_inputs)
    pos_ok = (all(c["positive"]["pass"] and c["positive"]["pass_original_definition"] for c in decision["controls"].values())
              if decision["controls"] else None)
    neg_ok = all(m["negative_control"]["pass"] for m in decision["models"].values())
    if n_conc >= 2:
        verdict, why = "no-go (concentrated)", f"top 10% of heads hold >= 90% of held-out jump mass on {n_conc} of 3 models"
    elif n_lex >= 2:
        verdict, why = "go", f"lexical >= 40% and >= 5x random outside the top 10% on {n_lex} of 3 models"
    else:
        verdict, why = "no-go", f"concentrated on {n_conc}, lexical passes on {n_lex} of 3 models"
    alt = "go" if n_conc < 2 and n_lex_norm >= 2 else verdict if n_conc >= 2 else "no-go"
    decision["verdict"] = {"verdict": verdict, "why": why, "models_concentrated": n_conc, "models_lexical_pass": n_lex,
                           "normalised_text_would_give": alt, "positive_controls_pass": pos_ok,
                           "negative_controls_pass": neg_ok,
                           "valid": bool(pos_ok is not False and neg_ok),
                           "flag": None if pos_ok else "positive control FAILED under both definitions (docs/JUMPS.md)"}
    (args.out / "decision.json").write_text(json.dumps(decision, indent=2))
    print(f"\nVERDICT: {verdict} ({why}); normalised text would give: {alt}; controls: positive "
          f"{'PASS' if pos_ok else 'FAILED (both definitions)'}, negative {'PASS' if neg_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
