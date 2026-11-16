#!/usr/bin/env python
"""Phase 0, part D: decode with simulated tiered KV reads and record quality and reads.

Prefill runs normally at fp16 (SDPA). Every decode step then goes through a patched
attention function that, for the new token, computes attention as a variant would
read the cache. Results are written per prompt and variant; analyze_d.py summarises.

Variants (docs/PHASE0.md, D):

    fp16            exact attention (the reference; same fp32 code path as the others)
    uniform8/4      every full block read from the 8-/4-bit copy
    hot_only        fresh oracle HOT set at fp16, every other block dropped
    fresh           (a) oracle plan recomputed every step from exact weights
    stale_R{R}      (b) oracle plan recomputed every R steps, reused unchanged between
    promote_R{R}    (c) stale plan + same-step promotion: rule B on this step's coarse
                    scores (fp16 for HOT, 4-bit keys for WARM, 1-bit keys for COLD);
                    blocks it marks HOT that the plan did not are read at fp16
    twotier_R{R}    (d) as (c) with no COLD tier (eps_c = 0), so only WARM is scanned

Simulation choices:
- Plans are refreshed from the exact fp16 weights of the refresh step (an oracle
  refresh). The refresh's read cost is charged in analyze_d.py as a 4-bit key scan of
  COLD blocks every R steps, per PHASE0 E.
- Blocks written after a refresh are HOT until the next refresh. Block 0 and the last
  two blocks at refresh time are HOT. The newest, partial block is therefore always
  fp16, in every variant including uniform8/4.
- Quantisers: uniform min-max with nearest rounding, keys per channel within each
  16-token block, values per token. Scales and minima are stored in fp16. 1-bit keys
  use the same quantiser with 2 levels.
- One-pass simulation of the two-pass read: HOT tokens use fp16 K/V, WARM tokens the
  dequantised copy, COLD tokens are masked. This equals the log-sum-exp merge of the
  partial states up to rounding.
- Attention for the new token is computed in fp32 for all variants, including fp16.

Quality is measured two ways per variant:
- teacher-forced over a reference continuation (the fp16 greedy output by default,
  or the record's reference answer with --reference answer): NLL per token, argmax
  agreement with the reference, per-layer relative L2 error of the attention output
  against exact attention computed from the same query and cache (a local error, on
  the variant's own hidden state);
- optionally free-running greedy (--free-run) for task scores.

Output: <out>/meta.json, <out>/done.jsonl, <out>/<prompt_id>/{prompt.json,<variant>.npz}.
Per-step arrays in <variant>.npz (T teacher-forced steps, L layers):
    nll [T], agree [T], err_num/err_den [T, L]   (sum over heads of |o - o_ref|^2, |o_ref|^2)
    counts [T, L, 6]  blocks summed over KV heads: planned HOT, planned WARM, planned
                      COLD, promoted from WARM, promoted from COLD, total
    recall [T, L, 4]  fresh-HOT blocks that the plan had COLD, of those promoted,
                      fresh-HOT blocks the plan had WARM, of those promoted
    tail [T, L, 8]    per query head true mass in COLD blocks before promotion: sum, count
                      > 0.02, > 0.05, > 0.10; then the same after promotion
    free_ids          free-running greedy tokens (with --free-run)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import COLD, HOT, WARM, Tiering  # noqa: E402
from capture import check_keys_cached_post_rope, code_version, encode, load_prompts, pick_device, sync  # noqa: E402

BLOCK = 16
ATTN_IMPL = "tiered_sdpa"
TAIL_THRESHOLDS = (0.02, 0.05, 0.10)


# --------------------------------------------------------------------------- variants


@dataclass(frozen=True)
class Variant:
    kind: str  # fp16 | uniform | hot_only | fresh | stale | promote | twotier
    bits: int = 4  # coarse bits (WARM, or everything for uniform)
    R: int = 1
    eps_w: float = 0.0
    eps_c: float = 0.0
    scan_bits: int = 1
    scan_codec: str = "minmax"

    @property
    def name(self) -> str:
        if self.kind == "fp16":
            return "fp16"
        if self.kind == "uniform":
            return f"uniform{self.bits}"
        tag = f"w{self.eps_w:g}"
        if self.kind == "hot_only":
            return f"hot_only_{tag}"
        if self.kind != "twotier":
            tag += f"_c{self.eps_c:g}"
        r = "" if self.kind == "fresh" else f"_R{self.R}"
        scan = "_scan-meanabs" if self.kind == "promote" and self.scan_codec != "minmax" else ""
        return f"{self.kind}{r}_{tag}{scan}"

    @property
    def needs_plan(self) -> bool:
        return self.kind in ("hot_only", "fresh", "stale", "promote", "twotier")

    @property
    def promotes(self) -> bool:
        return self.kind in ("promote", "twotier")


def build_variants(names, eps_ws, eps_cs, warm_bits, scan_codecs=("minmax",)) -> list[Variant]:
    out = {}
    for spec in names:
        if spec == "fp16":
            vs = [Variant("fp16")]
        elif spec.startswith("uniform"):
            vs = [Variant("uniform", bits=int(spec[len("uniform"):]))]
        else:
            kind, _, r = spec.partition("_R")
            R = int(r) if r else 1
            if kind not in ("hot_only", "fresh", "stale", "promote", "twotier"):
                raise ValueError(f"unknown variant {spec}")
            vs = []
            for w in eps_ws:
                for c in ([0.0] if kind in ("hot_only", "twotier") else eps_cs):
                    for sc in (scan_codecs if kind == "promote" else ["minmax"]):
                        vs.append(Variant(kind, bits=warm_bits, R=R, eps_w=w, eps_c=c, scan_codec=sc))
        for v in vs:
            out[v.name] = v
    return list(out.values())


# --------------------------------------------------------------------------- quantised copies


def quantise(x: torch.Tensor, bits: int, dim: int, codec: str = "minmax"):
    """Quantiser along dim with fp16 scale and offset; dequantise as code * scale + lo.

    minmax: uniform min-max levels, nearest rounding (PHASE0 D).
    meanabs (1 bit only): sign around the mean, levels mean -/+ mean absolute deviation.
    """
    if codec == "meanabs":
        assert bits == 1
        mu = x.mean(dim, keepdim=True)
        dev = (x - mu).abs().mean(dim, keepdim=True)
        lo, scale = (mu - dev).half(), (2 * dev).half()
        return (x > mu).to(torch.uint8), scale, lo
    lo = x.amin(dim, keepdim=True).half().float()
    hi = x.amax(dim, keepdim=True)
    levels = 2**bits - 1
    scale = ((hi - lo) / levels).half().float()
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    code = torch.where(scale > 0, ((x - lo) / safe).round(), torch.zeros_like(x)).clamp_(0, levels)
    return code.to(torch.uint8), scale.half(), lo.half()


def dequantise(code, scale, lo):
    return code.float() * scale.float() + lo.float()


class CoarseStore:
    """Quantised copies of a layer's full blocks, extended as blocks fill up."""

    def __init__(self, bits: int, values: bool, codec: str = "minmax"):
        self.bits, self.values, self.codec = bits, values, codec
        self.k = None  # (code, scale, lo) with code [H, nbf, 16, D]
        self.v = None
        self.nbf = 0

    def update(self, K: torch.Tensor, V: torch.Tensor):
        nbf = K.shape[1] // BLOCK
        if nbf <= self.nbf:
            return
        h, _, d = K.shape
        sl = slice(self.nbf * BLOCK, nbf * BLOCK)
        kb = K[:, sl].float().view(h, nbf - self.nbf, BLOCK, d)
        new_k = quantise(kb, self.bits, dim=2, codec=self.codec)  # per channel within the block
        self.k = new_k if self.k is None else tuple(torch.cat([a, b], 1) for a, b in zip(self.k, new_k))
        if self.values:
            vb = V[:, sl].float().view(h, nbf - self.nbf, BLOCK, d)
            new_v = quantise(vb, self.bits, dim=3)  # per token
            self.v = new_v if self.v is None else tuple(torch.cat([a, b], 1) for a, b in zip(self.v, new_v))
        self.nbf = nbf

    def full(self, which: str, fp: torch.Tensor) -> torch.Tensor:
        """Dequantised copy over all n tokens; tokens past the last full block stay fp16."""
        h, n, d = fp.shape
        parts = self.k if which == "k" else self.v
        dq = dequantise(*parts).view(h, self.nbf * BLOCK, d) if self.nbf else fp.new_zeros(h, 0, d).float()
        return torch.cat([dq, fp[:, self.nbf * BLOCK :].float()], 1)


# --------------------------------------------------------------------------- attention hook


def block_stats(w: torch.Tensor, nb: int):
    """w [..., n] -> m, q [..., nb]."""
    n = w.shape[-1]
    wb = F.pad(w, (0, nb * BLOCK - n)).view(*w.shape[:-1], nb, BLOCK)
    return wb.sum(-1), (wb * wb).sum(-1)


def rule_b(w: torch.Tensor, nb: int, eps_w: float, eps_c: float, hot_only: bool = False) -> torch.Tensor:
    """Rule B per query head, then max tier per KV head. w [H, G, n] -> [H, nb] int8."""
    m, q = block_stats(w, nb)
    t = Tiering(m, q).tiers(eps_w, eps_c)
    return t.amax(1)


class TieredState:
    def __init__(self, n_layers: int):
        self.L = n_layers
        self.active = False
        self.variant: Variant | None = None
        self.step = 0
        self.layers: dict[int, dict] = {}

    def reset(self, variant: Variant, device):
        self.variant = variant
        self.step = 0
        self.layers = {}
        z = lambda *s: torch.zeros(s, dtype=torch.float32, device=device)
        self.err = z(self.L, 2)
        self.counts = z(self.L, 6)
        self.recall = z(self.L, 4)
        self.tail = z(self.L, 8)

    def begin_step(self):
        for t in (self.err, self.counts, self.recall, self.tail):
            t.zero_()

    def end_step(self):
        out = torch.cat([self.err.flatten(), self.counts.flatten(), self.recall.flatten(), self.tail.flatten()]).cpu()
        self.step += 1
        L = self.L
        e, c, r, t = torch.split(out, [2 * L, 6 * L, 4 * L, 8 * L])
        return e.view(L, 2).numpy(), c.view(L, 6).numpy(), r.view(L, 4).numpy(), t.view(L, 8).numpy()


STATE = TieredState(0)


def _tail(mass: torch.Tensor) -> torch.Tensor:
    """mass [H, G] per query head -> [sum, count > thresholds...]."""
    return torch.stack([mass.sum()] + [(mass > th).sum().float() for th in TAIL_THRESHOLDS])


def tiered_attention(module, query, key, value, attention_mask, scaling):
    v = STATE.variant
    layer = module.layer_idx
    st = STATE.layers.setdefault(layer, {})
    _, hq, _, d = query.shape
    K, V = key[0], value[0]  # [H, n, D] fp16, post-RoPE keys
    h, n, _ = K.shape
    g = hq // h
    nb = math.ceil(n / BLOCK)
    scale = scaling if scaling is not None else d**-0.5
    qv = query[0, :, 0, :].float().view(h, g, d)
    Kf, Vf = K.float(), V.float()
    s16 = torch.bmm(qv, Kf.transpose(1, 2)) * scale  # [H, G, n]
    w16 = torch.softmax(s16, -1)
    o_ref = torch.bmm(w16, Vf)

    def attend(Ke, Ve, tok_mask=None):
        s = torch.bmm(qv, Ke.transpose(1, 2)) * scale
        if tok_mask is not None:
            s = s.masked_fill(tok_mask[:, None, :], float("-inf"))
        return torch.bmm(torch.softmax(s, -1), Ve)

    nbf = n // BLOCK
    counts = torch.zeros(6, device=K.device)
    counts[5] = h * nb
    if v.kind == "fp16":
        o = o_ref
        counts[0] = h * nb
    elif v.kind == "uniform":
        store = st.setdefault("coarse", CoarseStore(v.bits, values=True))
        store.update(K, V)
        o = attend(store.full("k", K), store.full("v", V))
        counts[0], counts[1] = h * (nb - nbf), h * nbf
    else:
        fresh = rule_b(w16, nb, v.eps_w, v.eps_c)
        if v.kind in ("fresh", "hot_only"):
            tiers = fresh
        else:
            if "plan" not in st or STATE.step % v.R == 0:
                st["plan"] = fresh
            plan = st["plan"]
            tiers = torch.cat([plan, plan.new_full((h, nb - plan.shape[1]), HOT)], 1)
        if v.kind == "hot_only":
            tiers = torch.where(tiers == HOT, tiers, torch.full_like(tiers, COLD))
        planned = tiers
        counts[0] = (planned == HOT).sum()
        counts[1] = (planned == WARM).sum()
        counts[2] = (planned == COLD).sum()
        tok = lambda t: t.repeat_interleave(BLOCK, dim=1)[:, :n]  # block tiers -> token tiers
        warm_store = None
        if v.kind != "hot_only":
            warm_store = st.setdefault("coarse", CoarseStore(v.bits, values=True))
            warm_store.update(K, V)
        if v.promotes:
            Kw = warm_store.full("k", K)
            s_c = torch.where(tok(planned == HOT)[:, None, :], s16, torch.bmm(qv, Kw.transpose(1, 2)) * scale)
            if v.kind == "promote":
                scan = st.setdefault("scan", CoarseStore(v.scan_bits, values=False, codec=v.scan_codec))
                scan.update(K, V)
                s_cold = torch.bmm(qv, scan.full("k", K).transpose(1, 2)) * scale
                s_c = torch.where(tok(planned == COLD)[:, None, :], s_cold, s_c)
            hot_c = rule_b(torch.softmax(s_c, -1), nb, v.eps_w, v.eps_c) == HOT
            promoted = hot_c & (planned != HOT)
            counts[3] = (promoted & (planned == WARM)).sum()
            counts[4] = (promoted & (planned == COLD)).sum()
            tiers = torch.where(promoted, torch.full_like(planned, HOT), planned)
        else:
            promoted = torch.zeros_like(planned, dtype=torch.bool)
        if v.kind not in ("fresh", "hot_only"):
            fresh_hot = fresh == HOT
            ev_c, ev_w = fresh_hot & (planned == COLD), fresh_hot & (planned == WARM)
            STATE.recall[layer] = torch.stack([ev_c.sum(), (ev_c & promoted).sum(), ev_w.sum(), (ev_w & promoted).sum()]).float()
        m16, _ = block_stats(w16, nb)  # [H, G, nb] true mass
        before = (m16 * (planned == COLD)[:, None, :]).sum(-1)
        after = (m16 * (tiers == COLD)[:, None, :]).sum(-1)
        STATE.tail[layer] = torch.cat([_tail(before), _tail(after)])
        hot_tok = tok(tiers == HOT)[:, :, None]
        if warm_store is None:
            Ke, Ve = Kf, Vf
        else:
            Ke = torch.where(hot_tok, Kf, warm_store.full("k", K))
            Ve = torch.where(hot_tok, Vf, warm_store.full("v", V))
        o = attend(Ke, Ve, tok(tiers == COLD))
    STATE.counts[layer] = counts
    STATE.err[layer, 0] = ((o - o_ref) ** 2).sum()
    STATE.err[layer, 1] = (o_ref**2).sum()
    return o.reshape(1, 1, hq, d).to(query.dtype)


def tiered_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    if STATE.active and query.shape[2] == 1:
        if query.shape[0] != 1:
            raise NotImplementedError("batch size 1 only")
        if kwargs.get("sliding_window") is not None and kwargs["sliding_window"] < key.shape[2]:
            raise NotImplementedError("sliding-window layers are not supported")
        if attention_mask is not None and attention_mask.dtype == torch.bool and not attention_mask[0, 0, -1, : key.shape[2]].all():
            raise NotImplementedError("masked keys in decode are not supported")
        return tiered_attention(module, query, key, value, attention_mask, scaling), None
    return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)


AttentionInterface.register(ATTN_IMPL, tiered_sdpa)
AttentionMaskInterface.register(ATTN_IMPL, sdpa_mask)


# --------------------------------------------------------------------------- decoding


@torch.inference_mode()
def decode(model, cache, n0, first_tok, device, *, teacher=None, max_new=0, eos_ids=frozenset()):
    """Teacher-forced over `teacher` (list of ids, teacher[0] = first_tok) or free-running.

    Returns dict of per-step arrays (teacher mode) or generated ids (free mode)."""
    cache.crop(n0)
    STATE.reset(STATE.variant, device)
    STATE.active = True
    rows = {k: [] for k in ("nll", "agree", "err", "counts", "recall", "tail")}
    gen = [first_tok]
    tok_in = first_tok
    steps = len(teacher) - 1 if teacher is not None else max_new - 1
    try:
        for t in range(steps):
            if teacher is None and tok_in in eos_ids:
                break
            STATE.begin_step()
            out = model(
                input_ids=torch.tensor([[tok_in]], device=device),
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor([n0 + t], device=device),
            )
            logits = out.logits[0, -1].float()
            e, c, r, tl = STATE.end_step()
            nxt = int(logits.argmax())
            if teacher is not None:
                target = teacher[t + 1]
                rows["nll"].append(float(-torch.log_softmax(logits, -1)[target]))
                rows["agree"].append(nxt == target)
                for k, x in zip(("err", "counts", "recall", "tail"), (e, c, r, tl)):
                    rows[k].append(x)
                tok_in = target
            else:
                gen.append(nxt)
                tok_in = nxt
    finally:
        STATE.active = False
    if teacher is None:
        return gen
    arr = {k: np.asarray(v, dtype=np.float32) for k, v in rows.items()}
    arr["agree"] = arr["agree"].astype(bool)
    arr["err_num"], arr["err_den"] = arr["err"][..., 0], arr["err"][..., 1]
    del arr["err"]
    return arr


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--prompts", type=Path, required=True, help="JSONL from make_needle_prompts.py or benchmarks.py")
    p.add_argument("--out", type=Path, required=True, help="output directory (under runs/)")
    p.add_argument("--max-context", type=int, required=True)
    p.add_argument("--max-new-tokens", type=int, default=128, help="cap; a record's own max_new_tokens is used if smaller")
    p.add_argument("--variants", nargs="+", default=[
        "fp16", "uniform8", "uniform4", "hot_only", "fresh",
        "stale_R4", "stale_R8", "promote_R4", "promote_R8", "twotier_R4", "twotier_R8"])
    p.add_argument("--eps-w", type=float, nargs="+", default=[1e-3])
    p.add_argument("--eps-c", type=float, nargs="+", default=[0.02])
    p.add_argument("--warm-bits", type=int, default=4)
    p.add_argument("--scan-codec", nargs="+", default=["minmax"], choices=["minmax", "meanabs"],
                   help="1-bit COLD key plane codec(s) for promote variants")
    p.add_argument("--reference", default="greedy", choices=["greedy", "answer"],
                   help="teacher-forced reference: the fp16 greedy output, or the record's first answer")
    p.add_argument("--free-run", action="store_true", help="also decode freely under every variant (task scores)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--chat-template", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--prefill-chunk", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    variants = build_variants(args.variants, args.eps_w, args.eps_c, args.warm_bits, args.scan_codec)
    if variants[0].kind != "fp16":
        variants = [Variant("fp16")] + [v for v in variants if v.kind != "fp16"]

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=getattr(torch, args.dtype), attn_implementation=ATTN_IMPL
    ).to(device).eval()
    cfg = model.config
    STATE.L = cfg.num_hidden_layers
    use_template = args.chat_template == "on" or (args.chat_template == "auto" and tok.chat_template is not None)
    eos = set()
    for e in (cfg.eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
        if e is not None:
            eos |= set(e if isinstance(e, list) else [e])

    rope = check_keys_cached_post_rope(model, device, start_pos=max(args.max_context, 4096))
    print(f"RoPE check: {rope}")
    if not rope["post_rope"]:
        sys.exit("cached keys do not look post-RoPE")

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "variants": [v.__dict__ | {"name": v.name} for v in variants],
        "seed": args.seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "model": args.model,
        "model_revision": getattr(cfg, "_commit_hash", None) or args.revision,
        "model_shape": {"layers": cfg.num_hidden_layers, "q_heads": cfg.num_attention_heads,
                        "kv_heads": cfg.num_key_value_heads, "head_dim": getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)},
        "block_size": BLOCK,
        "eos_ids": sorted(eos),
        "rope_check": rope,
        "code": code_version() | {"tiered_script": str(Path(__file__).resolve())},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    done_path = args.out / "done.jsonl"
    done = set()
    if done_path.exists() and not args.overwrite:
        done = {json.loads(l)["id"] for l in done_path.read_text().splitlines() if l.strip()}

    prompts = load_prompts(args.prompts, args.limit)
    print(f"device={device} variants={[v.name for v in variants]} prompts={len(prompts)}")
    for i, rec in enumerate(prompts):
        pid = str(rec["id"])
        if pid in done:
            continue
        ids, truncated = encode(tok, rec, use_template, args.max_context)
        n0 = len(ids)
        max_new = min(args.max_new_tokens, int(rec.get("max_new_tokens", args.max_new_tokens)))
        pdir = args.out / pid
        pdir.mkdir(exist_ok=True)
        cache = DynamicCache()
        STATE.active = False
        input_ids = torch.tensor([ids], device=device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            for s in range(0, n0, args.prefill_chunk):
                chunk = input_ids[:, s : s + args.prefill_chunk]
                out = model(input_ids=chunk, past_key_values=cache, use_cache=True,
                            cache_position=torch.arange(s, s + chunk.shape[1], device=device), logits_to_keep=1)
        first = int(out.logits[0, -1].argmax())
        sync(device)
        t_prefill = time.perf_counter() - t0

        STATE.variant = variants[0]  # fp16
        greedy = decode(model, cache, n0, first, device, max_new=max_new, eos_ids=eos)
        if args.reference == "answer":
            answer = (rec.get("answers") or [None])[0]
            if answer is None:
                sys.exit(f"{pid}: --reference answer needs an 'answers' field")
            teacher = [first] + tok(answer, add_special_tokens=False)["input_ids"][: max_new - 1]
        else:
            teacher = greedy
        timings = {"prefill_s": round(t_prefill, 2)}
        for v in variants:
            t0 = time.perf_counter()
            STATE.variant = v
            arr = decode(model, cache, n0, first, device, teacher=teacher)
            if args.free_run:
                arr["free_ids"] = np.asarray(
                    greedy if v.kind == "fp16" else decode(model, cache, n0, first, device, max_new=max_new, eos_ids=eos), np.int64)
            np.savez(pdir / f"{v.name}.npz", **arr)
            sync(device)
            timings[v.name] = round(time.perf_counter() - t0, 2)
            ppl = math.exp(arr["nll"].mean()) if len(arr["nll"]) else float("nan")
            err = np.sqrt(arr["err_num"].sum(0) / np.maximum(arr["err_den"].sum(0), 1e-30)).mean() if len(arr["nll"]) else float("nan")
            print(f"  {pid} {v.name:28} ppl={ppl:8.4f} agree={arr['agree'].mean():.3f} attn_err={err:.4f} ({timings[v.name]}s)")
        prompt_info = {
            "id": pid, "prompt_len": n0, "truncated": truncated, "teacher_ids": teacher, "greedy_ids": greedy,
            "reference": args.reference, "max_new_tokens": max_new,
            **{k: rec[k] for k in ("task", "answers", "metric", "source", "all_classes") if k in rec},
        }
        (pdir / "prompt.json").write_text(json.dumps(prompt_info))
        with open(done_path, "a") as f:
            f.write(json.dumps({"id": pid, "prompt_len": n0, "teacher_len": len(teacher), "timings": timings}) + "\n")
        print(f"[{i + 1}/{len(prompts)}] {pid}: prompt={n0} teacher={len(teacher)} prefill={t_prefill:.1f}s")
        del cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
