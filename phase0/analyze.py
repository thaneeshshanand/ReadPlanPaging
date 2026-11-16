#!/usr/bin/env python
"""Phase 0, parts B, C and E: tier fractions, read fraction rho and plan stability.

Reads the per-block statistics written by capture.py and, for every captured
pre-EOS decode step, layer and KV head, applies the tiering rule of
docs/PHASE0.md, B in two variants:

    exact        tier every query head on its own m_b, q_b, then give the KV head the
                 highest tier any of its query heads asks for (rule B as written)
    max_reduced  take the max of m_b and of q_b over a KV head's query heads, then
                 apply the rule once per KV head

for every eps_w x eps_c in the sweep. Tiers: COLD = 0, WARM = 1, HOT = 2. Block 0
and the last two blocks are always HOT.

Stability (part C): the plan computed at step t is applied unchanged at step t+d
for d = 1..16, extended with HOT for blocks written after step t (the tail blocks,
PHASE0 / proposal 5.2). Only pairs where both steps were captured and are pre-EOS
count. For refresh period R the per-R numbers average lags 1..R
("apply it unchanged at steps t+1 to t+R"). Metrics, per query head unless noted:

    stale_cold_mass   attention mass of the step t+d query landing in blocks the stale
                      plan marks COLD
    jaccard_hot       |HOT_stale & HOT_fresh| / |HOT_stale | HOT_fresh| per KV head
    missed            blocks HOT under the fresh plan but COLD under the stale one,
                      per KV head (count, fraction of fresh HOT, and the mass the query
                      heads put on them)

Read fraction (part E): rho = HOT + WARM * c/16 + COLD * 0.5 * (c/16) / R, with
block fractions averaged over steps, layers and KV heads, for c in {4, 8}.

Post-EOS steps are excluded: step t is the forward pass for query token gen_ids[t],
so steps at or after the first EOS in gen_ids are dropped.

Outputs in --out:
    summary.csv     one row per variant and (eps_w, eps_c): tier fractions, fresh COLD
                    mass, rho for c in {4, 8} and R in {1, 2, 4, 8, 16}
    by_R.csv        one row per variant, config and R: rho and stability averaged over
                    lags 1..R, with the go/no-go screen on the criteria this script
                    can check (rho <= 0.30, HOT <= 20%; perplexity is part D)
    stability.csv   one row per variant, config and lag d = 1..16
    per_prompt.csv  tier fractions per prompt, variant and config
    heatmap.npz     tier fractions per variant, config, layer and KV head
    meta.json       inputs, capture metadata, git commit, step counts
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

COLD, WARM, HOT = 0, 1, 2
EPS_W = [1e-2, 1e-3, 1e-4]
EPS_C = [0.005, 0.02, 0.05]
R_VALUES = [1, 2, 4, 8, 16]
MAX_LAG = 16
CODEC_BITS = [4, 8]
GO = {"rho": 0.30, "hot": 0.20, "R": 4}


# --------------------------------------------------------------------------- tiering rule


class Tiering:
    """Rule B on a batch of heads, with sort orders shared across the eps sweep.

    m, q: [..., nb] float tensors (one row per head). Rows are independent.
    """

    def __init__(self, m: torch.Tensor, q: torch.Tensor):
        nb = m.shape[-1]
        self.m, self.q, self.nb = m, q, nb
        forced = torch.zeros(nb, dtype=torch.bool, device=m.device)
        forced[0] = True
        forced[max(nb - 2, 0):] = True
        self.n_forced = int(forced.sum())
        key = q.masked_fill(forced, float("inf"))
        self.order_q = torch.argsort(key, dim=-1, descending=True, stable=True)
        qs = torch.gather(q.masked_fill(forced, 0.0), -1, self.order_q)
        # suffix[k] = q left outside HOT if the first k sorted blocks are HOT
        self.suffix = qs.flip(-1).cumsum(-1).flip(-1)
        self.order_m = torch.argsort(m, dim=-1, stable=True)
        self.m_sorted = torch.gather(m, -1, self.order_m)
        self.ranks = torch.arange(nb, device=m.device)

    def tiers(self, eps_w: float, eps_c: float) -> torch.Tensor:
        # HOT: smallest k >= n_forced with suffix[k] <= eps_w (suffix[nb] = 0).
        ok = self.suffix <= eps_w
        trailing_ok = ok.flip(-1).to(torch.int8).cumprod(-1).sum(-1)  # robust to rounding
        k = (self.nb - trailing_ok).clamp_min(self.n_forced)
        hot_sorted = self.ranks < k[..., None]
        hot = torch.zeros_like(hot_sorted).scatter(-1, self.order_q, hot_sorted)
        # COLD: non-HOT blocks by ascending mass while cumulative mass stays below eps_c.
        hot_by_m = torch.gather(hot, -1, self.order_m)
        cum = (self.m_sorted * ~hot_by_m).cumsum(-1)
        cold_sorted = ~hot_by_m & (cum < eps_c)
        cold = torch.zeros_like(cold_sorted).scatter(-1, self.order_m, cold_sorted)
        t = torch.full(hot.shape, WARM, dtype=torch.int8, device=hot.device)
        t[cold] = COLD
        t[hot] = HOT
        return t


def tier_rule(m: torch.Tensor, q: torch.Tensor, eps_w: float, eps_c: float) -> torch.Tensor:
    """Rule B for one config. m, q: [..., nb]. Returns int8 tiers [..., nb]."""
    return Tiering(m, q).tiers(eps_w, eps_c)


def plans(m: torch.Tensor, q: torch.Tensor, variant: str, configs) -> torch.Tensor:
    """KV-head plans for every config. m, q: [L, H, G, nb] (or [L, H, nb] for max_reduced
    input already reduced). Returns int8 [C, L, H, nb]."""
    if variant == "exact":
        tg = Tiering(m, q)
        return torch.stack([tg.tiers(w, c).amax(dim=-2) for w, c in configs])
    if m.dim() == 4:
        m, q = m.amax(dim=-2), q.amax(dim=-2)
    tg = Tiering(m, q)
    return torch.stack([tg.tiers(w, c) for w, c in configs])


# --------------------------------------------------------------------------- data


def load_capture(path: Path, eos_ids: set[int]) -> dict:
    d = np.load(path)
    m, q = d["m"], d["q"]
    q_is_sqrt = bool(d["q_is_sqrt"]) if "q_is_sqrt" in d.files else False
    gen = d["gen_ids"].tolist()
    first_eos = next((i for i, t in enumerate(gen) if t in eos_ids), len(gen))
    steps = d["step"]
    keep = steps < first_eos
    return {
        "m": m[keep],
        "q": q[keep],
        "q_is_sqrt": q_is_sqrt,
        "n_ctx": d["n_ctx"][keep],
        "step": steps[keep],
        "per_query_head": m.ndim == 5,
        "n_captured": int(len(steps)),
        "n_post_eos": int((~keep).sum()),
        "first_eos": first_eos if first_eos < len(gen) else None,
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


# --------------------------------------------------------------------------- accumulation


class Acc:
    """Running sums for one variant across all configs."""

    def __init__(self, n_cfg, L, H):
        z = lambda *s: np.zeros(s, np.float64)
        self.frac = z(n_cfg, 3)  # sum over (step, layer, head) of tier fractions
        self.n_heads = 0  # number of (step, layer, KV head) samples
        self.heat = z(n_cfg, L, H, 3)
        self.n_steps = 0
        self.fresh_cold_sum = z(n_cfg)
        self.fresh_cold_max = z(n_cfg)
        self.n_qheads = 0
        # stability, indexed [cfg, lag-1]
        self.pairs = z(MAX_LAG)
        self.sc_sum, self.sc_max = z(n_cfg, MAX_LAG), z(n_cfg, MAX_LAG)
        self.sc_gt_eps, self.sc_gt_10 = z(n_cfg, MAX_LAG), z(n_cfg, MAX_LAG)
        self.n_q_lag = z(MAX_LAG)
        self.jac_sum, self.n_kv_lag = z(n_cfg, MAX_LAG), z(MAX_LAG)
        self.missed_sum, self.fresh_hot_sum = z(n_cfg, MAX_LAG), z(n_cfg, MAX_LAG)
        self.missed_mass_sum = z(n_cfg, MAX_LAG)


def step_stats(acc: Acc, tiers: torch.Tensor, m_q: torch.Tensor):
    """Fresh-plan stats. tiers [C, L, H, nb]; m_q [L, H, G, nb] per query head mass."""
    nb = tiers.shape[-1]
    counts = torch.stack([(tiers == k).sum(-1) for k in (COLD, WARM, HOT)], -1).double() / nb  # [C, L, H, 3]
    acc.heat += counts.cpu().numpy()
    acc.frac += counts.sum((1, 2)).cpu().numpy()
    acc.n_heads += tiers.shape[1] * tiers.shape[2]
    acc.n_steps += 1
    cold_mass = torch.einsum("clhb,lhgb->clhg", (tiers == COLD).to(m_q.dtype), m_q)
    acc.fresh_cold_sum += cold_mass.sum((1, 2, 3)).cpu().numpy()
    acc.fresh_cold_max = np.maximum(acc.fresh_cold_max, cold_mass.amax((1, 2, 3)).cpu().numpy())
    acc.n_qheads += cold_mass[0].numel()


def lag_stats(acc: Acc, lag: int, stale: torch.Tensor, fresh: torch.Tensor, m_q: torch.Tensor, eps_c: torch.Tensor):
    """Stale plan (from step t-lag, padded with HOT) against the fresh plan at step t."""
    nb = fresh.shape[-1]
    pad = nb - stale.shape[-1]
    if pad:
        stale = torch.cat([stale, torch.full(stale.shape[:-1] + (pad,), HOT, dtype=stale.dtype, device=stale.device)], -1)
    i = lag - 1
    s_cold = (stale == COLD).to(m_q.dtype)
    cm = torch.einsum("clhb,lhgb->clhg", s_cold, m_q)  # [C, L, H, G]
    acc.pairs[i] += 1
    acc.n_q_lag[i] += cm[0].numel()
    acc.sc_sum[:, i] += cm.sum((1, 2, 3)).cpu().numpy()
    acc.sc_max[:, i] = np.maximum(acc.sc_max[:, i], cm.amax((1, 2, 3)).cpu().numpy())
    acc.sc_gt_eps[:, i] += (cm > eps_c[:, None, None, None]).sum((1, 2, 3)).cpu().numpy()
    acc.sc_gt_10[:, i] += (cm > 0.10).sum((1, 2, 3)).cpu().numpy()
    a, b = stale == HOT, fresh == HOT
    jac = (a & b).sum(-1).double() / (a | b).sum(-1).double()
    acc.jac_sum[:, i] += jac.sum((1, 2)).cpu().numpy()
    acc.n_kv_lag[i] += jac[0].numel()
    missed = b & (stale == COLD)
    acc.missed_sum[:, i] += missed.sum((1, 2, 3)).cpu().numpy()
    acc.fresh_hot_sum[:, i] += b.sum((1, 2, 3)).cpu().numpy()
    acc.missed_mass_sum[:, i] += torch.einsum("clhb,lhgb->c", missed.to(m_q.dtype), m_q).cpu().numpy()


def rho(fr, c, R):
    hot, warm, cold = fr[HOT], fr[WARM], fr[COLD]
    return hot + warm * c / 16 + cold * 0.5 * (c / 16) / R


# --------------------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captures", type=Path, nargs="+", required=True, help="capture output directories")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cpu", help="cpu or cuda (MPS lacks float64)")
    p.add_argument("--eps-w", type=float, nargs="+", default=EPS_W)
    p.add_argument("--eps-c", type=float, nargs="+", default=EPS_C)
    p.add_argument("--limit", type=int, default=None, help="only the first N prompts per capture directory")
    args = p.parse_args()
    device = torch.device(args.device)
    dt = torch.float64 if device.type != "mps" else torch.float32
    configs = [(w, c) for w in args.eps_w for c in args.eps_c]
    eps_c_t = torch.tensor([c for _, c in configs], dtype=dt, device=device)

    files, cap_metas = [], {}
    for d in args.captures:
        meta = json.loads((d / "meta.json").read_text())
        cap_metas[str(d)] = meta
        done = [json.loads(l) for l in (d / "done.jsonl").read_text().splitlines() if l.strip()]
        for rec in done[: args.limit]:
            files.append((d, rec["id"], set(meta["eos_ids"])))
    shapes = {(m["model_shape"]["layers"], m["model_shape"]["kv_heads"]) for m in cap_metas.values()}
    models = {m["model"] for m in cap_metas.values()}
    if len(shapes) != 1 or len(models) != 1:
        sys.exit(f"captures mix models: {models}")
    L, H = shapes.pop()

    variants = None
    accs = {}
    per_prompt = []
    n_steps_total = n_post_eos = 0
    for d, pid, eos_ids in files:
        cap = load_capture(d / f"{pid}.npz", eos_ids)
        v_here = ["exact", "max_reduced"] if cap["per_query_head"] else ["max_reduced"]
        if variants is None:
            variants = v_here
            accs = {v: Acc(len(configs), L, H) for v in variants}
        elif v_here != variants:
            sys.exit("captures mix per-query-head and max-reduced storage")
        n_steps_total += len(cap["step"])
        n_post_eos += cap["n_post_eos"]
        prompt_frac = {v: np.zeros((len(configs), 3)) for v in variants}
        prompt_n = 0
        history: dict[int, dict[str, torch.Tensor]] = {}
        for s in range(len(cap["step"])):
            t = int(cap["step"][s])
            nb = -(-int(cap["n_ctx"][s]) // 16)
            m = torch.from_numpy(cap["m"][s][..., :nb].astype(np.float64)).to(device, dt)
            q = torch.from_numpy(cap["q"][s][..., :nb].astype(np.float64)).to(device, dt)
            if cap["q_is_sqrt"]:
                q = q * q
            # per query head masses for the mass metrics; with max-reduced storage, the
            # KV-head max stands in (sums can exceed 1; flagged in meta.json)
            m_q = m if m.dim() == 4 else m[:, :, None, :]
            fresh = {}
            for v in variants:
                tiers = plans(m, q, v, configs)
                fresh[v] = tiers
                step_stats(accs[v], tiers, m_q)
                nbt = tiers.shape[-1]
                prompt_frac[v] += torch.stack([(tiers == k).double().mean(-1).mean((1, 2)) for k in (COLD, WARM, HOT)], -1).cpu().numpy()
                for lag in range(1, MAX_LAG + 1):
                    if t - lag in history:
                        lag_stats(accs[v], lag, history[t - lag][v], tiers, m_q, eps_c_t)
            prompt_n += 1
            history[t] = fresh
            for old in [k for k in history if k <= t - MAX_LAG]:
                del history[old]
        for v in variants:
            for ci, (w, c) in enumerate(configs):
                fr = prompt_frac[v][ci] / max(prompt_n, 1)
                per_prompt.append({"capture": str(d), "prompt": pid, "variant": v, "eps_w": w, "eps_c": c,
                                   "steps": prompt_n, "cold": fr[COLD], "warm": fr[WARM], "hot": fr[HOT]})
        print(f"{pid}: {prompt_n} pre-EOS steps ({cap['n_post_eos']} post-EOS dropped)")

    if not variants or n_steps_total == 0:
        sys.exit("no pre-EOS steps to analyse")

    args.out.mkdir(parents=True, exist_ok=True)
    summary, by_r, stab = [], [], []
    for v in variants:
        a = accs[v]
        for ci, (w, c) in enumerate(configs):
            fr = a.frac[ci] / a.n_heads
            row = {"variant": v, "eps_w": w, "eps_c": c, "samples": a.n_heads,
                   "hot": fr[HOT], "warm": fr[WARM], "cold": fr[COLD],
                   "fresh_cold_mass_mean": a.fresh_cold_sum[ci] / a.n_qheads, "fresh_cold_mass_max": a.fresh_cold_max[ci]}
            for cb in CODEC_BITS:
                for R in R_VALUES:
                    row[f"rho_c{cb}_R{R}"] = rho(fr, cb, R)
            summary.append(row)
            for lag in range(1, MAX_LAG + 1):
                i = lag - 1
                if a.pairs[i] == 0:
                    continue
                stab.append({
                    "variant": v, "eps_w": w, "eps_c": c, "lag": lag, "pairs": int(a.pairs[i]),
                    "stale_cold_mass_mean": a.sc_sum[ci, i] / a.n_q_lag[i],
                    "stale_cold_mass_max": a.sc_max[ci, i],
                    "frac_qheads_stale_cold_mass_gt_eps_c": a.sc_gt_eps[ci, i] / a.n_q_lag[i],
                    "frac_qheads_stale_cold_mass_gt_0.10": a.sc_gt_10[ci, i] / a.n_q_lag[i],
                    "jaccard_hot_mean": a.jac_sum[ci, i] / a.n_kv_lag[i],
                    "missed_blocks_per_kv_head": a.missed_sum[ci, i] / a.n_kv_lag[i],
                    "missed_frac_of_fresh_hot": a.missed_sum[ci, i] / max(a.fresh_hot_sum[ci, i], 1),
                    "missed_mass_per_query_head": a.missed_mass_sum[ci, i] / a.n_q_lag[i],
                })
            for R in R_VALUES:
                lags = [i for i in range(R) if a.pairs[i] > 0]
                complete = len(lags) == R
                agg = lambda num, den: (sum(num[ci, i] for i in lags) / sum(den[i] for i in lags)) if lags else float("nan")
                for cb in CODEC_BITS:
                    r = rho(fr, cb, R)
                    by_r.append({
                        "variant": v, "eps_w": w, "eps_c": c, "R": R, "codec_bits": cb, "rho": r, "hot": fr[HOT],
                        "lags_with_data": len(lags),
                        "stale_cold_mass_mean": agg(a.sc_sum, a.n_q_lag),
                        "stale_cold_mass_max": max((a.sc_max[ci, i] for i in lags), default=float("nan")),
                        "jaccard_hot_mean": agg(a.jac_sum, a.n_kv_lag),
                        "missed_blocks_per_kv_head": agg(a.missed_sum, a.n_kv_lag),
                        "screen_rho_hot_R": bool(r <= GO["rho"] and fr[HOT] <= GO["hot"] and R >= GO["R"] and complete),
                    })

    def write(name, rows):
        with open(args.out / name, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            for r in rows:
                wr.writerow({k: (f"{x:.6g}" if isinstance(x, float) else x) for k, x in r.items()})

    write("summary.csv", summary)
    write("by_R.csv", by_r)
    if stab:
        write("stability.csv", stab)
    write("per_prompt.csv", per_prompt)
    np.savez(
        args.out / "heatmap.npz",
        fractions=np.stack([accs[v].heat / accs[v].n_steps for v in variants]),  # [V, C, L, H, 3]
        variants=np.array(variants), eps_w=np.array([w for w, _ in configs]), eps_c=np.array([c for _, c in configs]),
        tier_order=np.array(["COLD", "WARM", "HOT"]),
    )
    here = Path(__file__).resolve()
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: [str(x) for x in v] if isinstance(v, list) and v and isinstance(v[0], Path) else (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "git_commit": git_commit(),
        "script_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
        "variants": variants,
        "mass_metrics_per_query_head": variants[0] == "exact",
        "prompts": len(files),
        "pre_eos_steps": n_steps_total,
        "post_eos_steps_dropped": n_post_eos,
        "go_criteria_checked_here": GO,
        "captures": cap_metas,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n{len(files)} prompts, {n_steps_total} pre-EOS steps, {n_post_eos} post-EOS steps dropped -> {args.out}")
    print(f"{'variant':12} {'eps_w':>7} {'eps_c':>6} {'HOT':>6} {'WARM':>6} {'COLD':>6} {'rho4,R4':>8} {'staleCold@R4':>12} {'Jacc@R4':>8}")
    for row in summary:
        b = next(x for x in by_r if x["variant"] == row["variant"] and x["eps_w"] == row["eps_w"]
                 and x["eps_c"] == row["eps_c"] and x["R"] == 4 and x["codec_bits"] == 4)
        print(f"{row['variant']:12} {row['eps_w']:7.0e} {row['eps_c']:6.3f} {row['hot']:6.3f} {row['warm']:6.3f} "
              f"{row['cold']:6.3f} {row['rho_c4_R4']:8.3f} {b['stale_cold_mass_mean']:12.4f} {b['jaccard_hot_mean']:8.3f}")


if __name__ == "__main__":
    main()
