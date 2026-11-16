#!/usr/bin/env python
"""Phase 0, part D: decode with simulated tiered KV reads and record quality and reads.

Prefill runs normally at fp16 (SDPA). Every decode step then goes through a patched
attention function that, for the new token, computes attention as a variant would
read the cache. Results are written per prompt and variant; analyze_d.py summarises.

Variants (docs/PHASE0.md, D):

    fp16            exact attention (the reference; same fp32 code path as the others)
    fp8             every key and value cast to emulated fp8 e4m3, scale 1, saturating
    uniform8/4/3/2  every full block read from the 8-/4-/3-/2-bit copy
    uniformK4V3     asymmetric uniform: keys at 4 bits, values at 3 (any K{k}V{v})
    hot_only        fresh oracle HOT set at fp16, every other block dropped
    fresh           (a) oracle plan recomputed every step from exact weights
    stale_R{R}      (b) oracle plan recomputed every R steps, reused unchanged between
    promote_R{R}    (c) stale plan + same-step promotion: rule B on this step's coarse
                    scores (fp16 for HOT, 4-bit keys for WARM, 1-bit keys for COLD);
                    blocks it marks HOT that the plan did not are read at fp16
    twotier_R{R}    (d) as (c) with no COLD tier (eps_c = 0), so only WARM is scanned
    coldkeys_R{R}   (e) as (c), but COLD blocks have their 4-bit keys read every step
                    (values skipped), so promotion scores them like WARM blocks
    live            (f) no stored plan: every step reads all keys at the WARM width, applies
                    rule B to those scores, then reads values by tier (HOT: keys and values
                    at the HOT width; WARM: values at the WARM width; COLD: no values).
                    No oracle refresh, no fix-ups.

Re-based ladder (--hot-bits b*, --warm-bits w): HOT blocks, fix-up reads and the HOT
scores used for promotion come from a b*-bit copy instead of fp16 (the newest, partial
block stays fp16); WARM uses w bits.

Key codecs (--key-codecs) set how key scales and offsets are shared, for uniform and
tiered variants alike: per 16-token block (block), per 256-token group (group256), or
static per layer/head/channel values from calibrate_keys.py (static, --key-calib).

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
--reference record uses a record's reference_ids (continuation sets from benchmarks.py).

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


TIERED_KINDS = ("hot_only", "fresh", "stale", "promote", "twotier", "coldkeys", "live")
KEY_CODECS = ("block", "group256", "static")


@dataclass(frozen=True)
class Variant:
    kind: str  # fp16 | fp8 | uniform | hot_only | fresh | stale | promote | twotier | coldkeys | live
    bits: int = 4  # coarse bits (WARM, or everything for uniform; keys only when vbits is set)
    R: int = 1
    eps_w: float = 0.0
    eps_c: float = 0.0
    scan_bits: int = 1
    scan_codec: str = "minmax"
    kcodec: str = "block"  # key scale/offset sharing: per 16-token block, per 256-token group, static
    hot_bits: int = 16  # HOT tier precision: 16 = fp16, else a b*-bit copy (re-based ladder)
    vbits: int | None = None  # uniform only: value bits when they differ from the key bits

    @property
    def value_bits(self) -> int:
        return self.bits if self.vbits is None else self.vbits

    @property
    def name(self) -> str:
        if self.kind in ("fp16", "fp8"):
            return self.kind
        ksuf = "" if self.kcodec == "block" else f"_k{self.kcodec}"
        if self.kind == "uniform":
            if self.value_bits != self.bits:
                return f"uniformK{self.bits}V{self.value_bits}{ksuf}"
            return f"uniform{self.bits}{ksuf}"
        tag = f"w{self.eps_w:g}"
        if self.kind == "hot_only":
            return f"hot_only_{tag}"
        if self.kind != "twotier":
            tag += f"_c{self.eps_c:g}"
        r = "" if self.kind in ("fresh", "live") else f"_R{self.R}"
        scan = "_scan-meanabs" if self.kind == "promote" and self.scan_codec != "minmax" else ""
        ladder = "" if self.hot_bits >= 16 else f"_h{self.hot_bits}b{self.bits}"
        return f"{self.kind}{r}_{tag}{scan}{ladder}{ksuf}"

    @property
    def promotes(self) -> bool:
        return self.kind in ("promote", "twotier", "coldkeys")


def build_variants(names, eps_ws, eps_cs, warm_bits, scan_codecs=("minmax",), kcodecs=("block",), hot_bits=16) -> list[Variant]:
    out = {}
    for spec in names:
        if spec in ("fp16", "fp8"):
            vs = [Variant(spec)]
        elif spec.startswith("uniformK"):
            kb, _, vb = spec[len("uniformK"):].partition("V")
            vs = [Variant("uniform", bits=int(kb), vbits=int(vb), kcodec=kc) for kc in kcodecs]
        elif spec.startswith("uniform"):
            vs = [Variant("uniform", bits=int(spec[len("uniform"):]), kcodec=kc) for kc in kcodecs]
        else:
            kind, _, r = spec.partition("_R")
            R = int(r) if r else 1
            if kind not in TIERED_KINDS:
                raise ValueError(f"unknown variant {spec}")
            vs = []
            for w in eps_ws:
                for c in ([0.0] if kind in ("hot_only", "twotier") else eps_cs):
                    for sc in (scan_codecs if kind == "promote" else ["minmax"]):
                        for kc in (["block"] if kind == "hot_only" else kcodecs):
                            vs.append(Variant(kind, bits=warm_bits, R=R, eps_w=w, eps_c=c, scan_codec=sc, kcodec=kc,
                                              hot_bits=hot_bits))
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


def quantise_fixed(x: torch.Tensor, bits: int, lo: torch.Tensor, scale: torch.Tensor, codec: str = "minmax"):
    """Quantise with given (broadcastable) offset and scale, e.g. static calibrated values.
    Values outside the calibrated range saturate."""
    lo, scale = lo.half().float(), scale.half().float()
    if codec == "meanabs":
        code = (x > lo + scale / 2).to(torch.uint8)
    else:
        levels = 2**bits - 1
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        code = ((x - lo) / safe).round().clamp_(0, levels).to(torch.uint8)
    shape = x.shape[:-2] + (1, x.shape[-1])
    return code, scale.expand(shape).half().contiguous(), lo.expand(shape).half().contiguous()


def dequantise(code, scale, lo):
    return code.float() * scale.float() + lo.float()


def fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Emulated cast to fp8 e4m3 (fn) with scale 1, saturating at +-448, nearest rounding.
    Works on any device; matches torch.float8_e4m3fn for in-range values."""
    a = x.abs().clamp(max=448.0)
    e = torch.floor(torch.log2(a.clamp(min=2.0**-9))).clamp(min=-6)
    step = torch.exp2(e - 3)
    return torch.sign(x) * (a / step).round() * step


GROUP_BLOCKS = 16  # 256-token key groups


class CoarseStore:
    """Quantised copies of a layer's full blocks, extended as blocks fill up.

    Key codecs (values are always per token):
      block     scale/offset per channel per 16-token block
      group256  scale/offset per channel shared by 16 consecutive blocks; blocks of a group
                that is not yet complete use per-block values until it completes
      static    per-layer, per-head, per-channel values calibrated offline (calib)
    """

    def __init__(self, bits: int, values: bool, codec: str = "minmax", kcodec: str = "block", calib: dict | None = None,
                 vbits: int | None = None):
        self.bits, self.values, self.codec, self.kcodec = bits, values, codec, kcodec
        self.vbits = bits if vbits is None else vbits
        self.k = None  # (code, scale, lo) with code [H, nbf, 16, D]
        self.v = None
        self.nbf = 0
        self.groups_done = 0
        if kcodec == "static":
            if calib is None:
                raise ValueError("static key codec needs calibration (--key-calib)")
            if codec == "meanabs":
                lo, sc = calib["mean"] - calib["mad"], 2 * calib["mad"]
            else:
                lo, sc = calib["min"], (calib["max"] - calib["min"]) / (2**bits - 1)
            self.static = (lo[:, None, None, :], sc[:, None, None, :])  # [H, 1, 1, D]

    def update(self, K: torch.Tensor, V: torch.Tensor):
        nbf = K.shape[1] // BLOCK
        if nbf <= self.nbf:
            return
        h, _, d = K.shape
        sl = slice(self.nbf * BLOCK, nbf * BLOCK)
        kb = K[:, sl].float().view(h, nbf - self.nbf, BLOCK, d)
        if self.kcodec == "static":
            lo, sc = (t.to(kb.device) for t in self.static)
            new_k = quantise_fixed(kb, self.bits, lo, sc, self.codec)
        else:
            new_k = quantise(kb, self.bits, dim=2, codec=self.codec)  # per channel within the block
        self.k = new_k if self.k is None else tuple(torch.cat([a, b], 1) for a, b in zip(self.k, new_k))
        if self.kcodec == "group256":
            gb = GROUP_BLOCKS
            while (self.groups_done + 1) * gb <= nbf:
                g0, g1 = self.groups_done * gb, (self.groups_done + 1) * gb
                kg = K[:, g0 * BLOCK : g1 * BLOCK].float().view(h, gb, BLOCK, d)
                c, sc, lo = quantise(kg, self.bits, dim=(1, 2), codec=self.codec)
                self.k[0][:, g0:g1] = c
                self.k[1][:, g0:g1] = sc.expand(h, gb, 1, d)
                self.k[2][:, g0:g1] = lo.expand(h, gb, 1, d)
                self.groups_done += 1
        if self.values:
            vb = V[:, sl].float().view(h, nbf - self.nbf, BLOCK, d)
            new_v = quantise(vb, self.vbits, dim=3)  # per token
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
        self.calib: dict | None = None  # layer -> {min, max, mean, mad} [H, D]
        self.verify = False
        self.verify_err = 0.0  # max relative L2 error of the patched fp16 output against stock SDPA

    def layer_calib(self, layer: int, device):
        if self.calib is None:
            return None
        return {k: t.to(device) for k, t in self.calib[layer].items()}

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
    elif v.kind == "fp8":
        o = attend(fp8_e4m3(Kf), fp8_e4m3(Vf))
        counts[1] = h * nb
    elif v.kind == "uniform":
        if "coarse" not in st:
            st["coarse"] = CoarseStore(v.bits, values=True, kcodec=v.kcodec, calib=STATE.layer_calib(layer, K.device),
                                       vbits=v.value_bits)
        store = st["coarse"]
        store.update(K, V)
        o = attend(store.full("k", K), store.full("v", V))
        counts[0], counts[1] = h * (nb - nbf), h * nbf
    else:
        fresh = rule_b(w16, nb, v.eps_w, v.eps_c)
        if v.kind == "live":  # (f): rule B on this step's WARM-width key scores, nothing stored
            if "coarse" not in st:
                st["coarse"] = CoarseStore(v.bits, values=True, kcodec=v.kcodec, calib=STATE.layer_calib(layer, K.device))
            st["coarse"].update(K, V)
            s_w = torch.bmm(qv, st["coarse"].full("k", K).transpose(1, 2)) * scale
            tiers = rule_b(torch.softmax(s_w, -1), nb, v.eps_w, v.eps_c)
        elif v.kind in ("fresh", "hot_only"):
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
        if v.hot_bits < 16:  # re-based ladder: HOT reads come from the b*-bit copy
            if "hot" not in st:
                st["hot"] = CoarseStore(v.hot_bits, values=True, kcodec=v.kcodec, calib=STATE.layer_calib(layer, K.device))
            st["hot"].update(K, V)
            Kh, Vh = st["hot"].full("k", K), st["hot"].full("v", V)
            s_hot = torch.bmm(qv, Kh.transpose(1, 2)) * scale
        else:
            Kh, Vh, s_hot = Kf, Vf, s16
        warm_store = None
        if v.kind != "hot_only":
            if "coarse" not in st:
                st["coarse"] = CoarseStore(v.bits, values=True, kcodec=v.kcodec, calib=STATE.layer_calib(layer, K.device))
            warm_store = st["coarse"]
            warm_store.update(K, V)
        if v.promotes:
            Kw = warm_store.full("k", K)
            s_c = torch.where(tok(planned == HOT)[:, None, :], s_hot, torch.bmm(qv, Kw.transpose(1, 2)) * scale)
            if v.kind == "promote":
                if "scan" not in st:
                    st["scan"] = CoarseStore(v.scan_bits, values=False, codec=v.scan_codec, kcodec=v.kcodec,
                                             calib=STATE.layer_calib(layer, K.device))
                scan = st["scan"]
                scan.update(K, V)
                s_cold = torch.bmm(qv, scan.full("k", K).transpose(1, 2)) * scale
                s_c = torch.where(tok(planned == COLD)[:, None, :], s_cold, s_c)
            # twotier has no COLD; coldkeys (e) scores COLD blocks with their 4-bit keys like WARM
            hot_c = rule_b(torch.softmax(s_c, -1), nb, v.eps_w, v.eps_c) == HOT
            promoted = hot_c & (planned != HOT)
            counts[3] = (promoted & (planned == WARM)).sum()
            counts[4] = (promoted & (planned == COLD)).sum()
            tiers = torch.where(promoted, torch.full_like(planned, HOT), planned)
        else:
            promoted = torch.zeros_like(planned, dtype=torch.bool)
        if v.kind == "live":  # agreement with the oracle plan: [oracle HOT, of those live HOT, live HOT, oracle HOT live COLD]
            fresh_hot, live_hot = fresh == HOT, planned == HOT
            STATE.recall[layer] = torch.stack([fresh_hot.sum(), (fresh_hot & live_hot).sum(), live_hot.sum(),
                                               (fresh_hot & (planned == COLD)).sum()]).float()
        elif v.kind not in ("fresh", "hot_only"):
            fresh_hot = fresh == HOT
            ev_c, ev_w = fresh_hot & (planned == COLD), fresh_hot & (planned == WARM)
            STATE.recall[layer] = torch.stack([ev_c.sum(), (ev_c & promoted).sum(), ev_w.sum(), (ev_w & promoted).sum()]).float()
        m16, _ = block_stats(w16, nb)  # [H, G, nb] true mass
        before = (m16 * (planned == COLD)[:, None, :]).sum(-1)
        after = (m16 * (tiers == COLD)[:, None, :]).sum(-1)
        STATE.tail[layer] = torch.cat([_tail(before), _tail(after)])
        hot_tok = tok(tiers == HOT)[:, :, None]
        if warm_store is None:
            Ke, Ve = Kh, Vh
        else:
            Ke = torch.where(hot_tok, Kh, warm_store.full("k", K))
            Ve = torch.where(hot_tok, Vh, warm_store.full("v", V))
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
        o = tiered_attention(module, query, key, value, attention_mask, scaling)
        if STATE.verify and STATE.variant.kind == "fp16":  # the patched fp16 path against stock SDPA
            ref = sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                         is_causal=is_causal, **kwargs)[0].float()
            err = ((o.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
            STATE.verify_err = max(STATE.verify_err, err)
        return o, None
    return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)


AttentionInterface.register(ATTN_IMPL, tiered_sdpa)
AttentionMaskInterface.register(ATTN_IMPL, sdpa_mask)


# --------------------------------------------------------------------------- decoding


@torch.inference_mode()
def decode(model, cache, n0, first_tok, device, *, teacher=None, max_new=0, eos_ids=frozenset(), stock=False):
    """Teacher-forced over `teacher` (list of ids, teacher[0] = first_tok) or free-running.
    stock=True bypasses the patched attention (stock SDPA), for --verify.

    Returns dict of per-step arrays (teacher mode) or generated ids (free mode)."""
    cache.crop(n0)
    STATE.reset(STATE.variant, device)
    STATE.active = not stock
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


def write_meta(path: Path, meta: dict):
    """Merge this invocation into meta.json: variant definitions accumulate (union by name)
    and every invocation keeps its own record, so adding variants never loses provenance."""
    invocations, vmap = [], {}
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("model") != meta["model"]:
            sys.exit(f"{path} belongs to {old.get('model')}, not {meta['model']}")
        invocations = old.get("invocations") or [
            {k: x for k, x in old.items() if k != "variants"} | {"variants": [v["name"] for v in old.get("variants", [])]}]
        vmap = {v["name"]: v for v in old.get("variants", [])}
    vmap.update({v["name"]: v for v in meta["variants"]})
    record = {k: x for k, x in meta.items() if k != "variants"} | {"variants": [v["name"] for v in meta["variants"]]}
    merged = meta | {"variants": list(vmap.values()), "invocations": invocations + [record]}
    path.write_text(json.dumps(merged, indent=2))


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
    p.add_argument("--hot-bits", type=int, default=16, help="HOT tier precision for the re-based ladder (16 = fp16)")
    p.add_argument("--scan-codec", nargs="+", default=["minmax"], choices=["minmax", "meanabs"],
                   help="1-bit COLD key plane codec(s) for promote variants")
    p.add_argument("--key-codecs", nargs="+", default=["block"], choices=list(KEY_CODECS))
    p.add_argument("--key-calib", type=Path, default=None, help="calibrate_keys.py output, for --key-codecs static")
    p.add_argument("--shard", default=None, help="i/n: run prompts i, i+n, i+2n, ... (Slurm arrays)")
    p.add_argument("--reference", default="greedy", choices=["greedy", "answer", "record"],
                   help="teacher-forced reference: fp16 greedy output, the record's first answer, or its reference_ids")
    p.add_argument("--free-run", action="store_true", help="also decode freely under every variant (task scores)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--chat-template", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--prefill-chunk", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verify", action="store_true",
                   help="per prompt: check the patched fp16 path against stock SDPA (attention output and teacher-forced "
                        "logits); exits if the attention error exceeds --verify-tol")
    p.add_argument("--verify-tol", type=float, default=1e-2)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    variants = build_variants(args.variants, args.eps_w, args.eps_c, args.warm_bits, args.scan_codec, args.key_codecs,
                              args.hot_bits)
    if variants[0].kind != "fp16":
        variants = [Variant("fp16")] + [v for v in variants if v.kind != "fp16"]

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=getattr(torch, args.dtype), attn_implementation=ATTN_IMPL
    ).to(device).eval()
    cfg = model.config
    STATE.L = cfg.num_hidden_layers
    calib_meta = None
    if any(v.kcodec == "static" for v in variants):
        if args.key_calib is None:
            sys.exit("--key-codecs static needs --key-calib")
        blob = torch.load(args.key_calib, map_location="cpu")
        if blob["model"] != args.model:
            sys.exit(f"calibration is for {blob['model']}, not {args.model}")
        STATE.calib = blob["layers"]
        calib_meta = {k: v for k, v in blob.items() if k != "layers"} | {"path": str(args.key_calib)}
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
        "key_calibration": calib_meta,
        "code": code_version() | {"tiered_script": str(Path(__file__).resolve())},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
    }
    write_meta(args.out / "meta.json", meta)

    done_path = args.out / "done.jsonl"
    done = set()
    if done_path.exists() and not args.overwrite:
        done = {json.loads(l)["id"] for l in done_path.read_text().splitlines() if l.strip()}

    prompts = load_prompts(args.prompts, args.limit)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        prompts = prompts[i::n]
    print(f"device={device} variants={[v.name for v in variants]} prompts={len(prompts)}")
    for i, rec in enumerate(prompts):
        pid = str(rec["id"])
        pdir = args.out / pid
        todo = [v for v in variants if args.overwrite or pid not in done or not (pdir / f"{v.name}.npz").exists()]
        if not todo:
            continue  # every requested variant already exists; new variants are added to finished prompts
        ids, truncated = encode(tok, rec, use_template, args.max_context)
        n0 = len(ids)
        max_new = min(args.max_new_tokens, int(rec.get("max_new_tokens", args.max_new_tokens)))
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
        need_greedy = args.reference == "greedy" or args.free_run
        greedy = decode(model, cache, n0, first, device, max_new=max_new, eos_ids=eos) if need_greedy else []
        if args.reference == "record":
            if "reference_ids" not in rec:
                sys.exit(f"{pid}: --reference record needs reference_ids")
            teacher = list(rec["reference_ids"])[:max_new]
        elif args.reference == "answer":
            answer = (rec.get("answers") or [None])[0]
            if answer is None:
                sys.exit(f"{pid}: --reference answer needs an 'answers' field")
            teacher = [first] + tok(answer, add_special_tokens=False)["input_ids"][: max_new - 1]
        else:
            teacher = greedy
        timings = {"prefill_s": round(t_prefill, 2)}
        verify = None
        if args.verify:
            STATE.variant, STATE.verify, STATE.verify_err = variants[0], True, 0.0
            patched = decode(model, cache, n0, teacher[0], device, teacher=teacher)
            STATE.verify = False
            stock = decode(model, cache, n0, teacher[0], device, teacher=teacher, stock=True)
            verify = {"attn_rel_err_max": STATE.verify_err,
                      "nll_abs_diff_max": float(np.abs(patched["nll"] - stock["nll"]).max()) if len(stock["nll"]) else 0.0,
                      "agree_patched": float(patched["agree"].mean()), "agree_stock": float(stock["agree"].mean()),
                      "ppl_patched": math.exp(patched["nll"].mean()), "ppl_stock": math.exp(stock["nll"].mean())}
            print(f"  verify {pid}: {verify}")
            if verify["attn_rel_err_max"] > args.verify_tol:
                sys.exit(f"--verify failed: patched fp16 attention differs from SDPA by {verify['attn_rel_err_max']:.3g}")
        for v in todo:
            t0 = time.perf_counter()
            STATE.variant = v
            arr = decode(model, cache, n0, teacher[0], device, teacher=teacher)
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
            f.write(json.dumps({"id": pid, "prompt_len": n0, "teacher_len": len(teacher), "timings": timings,
                                **({"verify": verify} if verify else {})}) + "\n")
        print(f"[{i + 1}/{len(prompts)}] {pid}: prompt={n0} teacher={len(teacher)} prefill={t_prefill:.1f}s")
        del cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
