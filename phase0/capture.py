#!/usr/bin/env python
"""Phase 0, part A: capture per-block attention statistics during greedy decode.

For every captured decode step, layer and KV head this writes two arrays over
16-token blocks of the context:

    m_b = sum of the new token's attention weights in block b
    q_b = sum of the squared attention weights in block b

Arrays are kept per query head by default, so analysis can apply tiering rule B
exactly (tier per query head, then the highest tier per KV head). Pass
--kv-reduce max to store only the max over each KV head's query heads; that
saves G x space but only supports the max-reduced variant (docs/PHASE0.md, A).

With --store-dtype float16, q holds sqrt(q_b) because q_b underflows fp16 at long
context; the npz field q_is_sqrt and meta.json say which is stored.

How it works: the model runs with transformers' normal SDPA attention, so the
generated text is unchanged. A registered attention function ("capture_sdpa")
sees the post-RoPE query and the full cached keys at each layer. When the query
is a single decode token it recomputes that token's softmax over the n cached
keys in fp32 (n values per query head, never an n x n matrix) and reduces it to
blocks. Prefill is chunked so long prompts never build a full attention matrix.

Analysis lives elsewhere: this script only writes statistics to disk.

Output directory layout:
    meta.json                 run config, seed, model revision, code version, checks
    <prompt_id>.npz           m, q, n_ctx, step, gen_ids, prompt_len
    done.jsonl                one line per finished prompt (timings, sizes)

Array shapes in <prompt_id>.npz (S captured steps, L layers, H KV heads,
G query heads per KV head, B blocks at the last step):
    m, q      [S, L, H, G, B]  or [S, L, H, B] with --kv-reduce max
    q_is_sqrt ()   True if q holds sqrt(q_b)
    n_ctx     [S]  number of keys the step attended over (includes the new token)
    step      [S]  decode-step index of each captured row
Blocks at index >= ceil(n_ctx/16) are zero padding. With the max reduction the
per-head sum of m over blocks is >= 1, not == 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask

BLOCK = 16
ATTN_IMPL = "capture_sdpa"


# --------------------------------------------------------------------------- capture hook


class CaptureState:
    def __init__(self):
        self.active = False
        self.per_query_head = False
        self.verify = False
        self.m = None  # device tensor [L, H, (G,) B_max]
        self.q = None
        self.verify_err = {}  # layer -> relative L2 error of w @ V against SDPA output

    def reset_step(self):
        self.m.zero_()
        self.q.zero_()


STATE = CaptureState()


def _new_token_weights(module, query, key, attention_mask, scaling, kwargs):
    """Attention weights of the single new query over all cached keys, fp32.

    query: [1, Hq, 1, D] post-RoPE. key: [1, H, n, D] post-RoPE, as cached.
    Returns w: [H, G, n].
    """
    if kwargs.get("sliding_window") is not None and kwargs["sliding_window"] < key.shape[2]:
        raise NotImplementedError("sliding-window layers are not supported by the capture")
    if kwargs.get("softcap") is not None:
        raise NotImplementedError("attention logit softcapping is not supported by the capture")
    _, hq, _, d = query.shape
    h, n = key.shape[1], key.shape[2]
    g = hq // h
    if scaling is None:
        scaling = d ** -0.5
    qv = query[0, :, 0, :].float().view(h, g, d)
    s = torch.bmm(qv, key[0].float().transpose(1, 2)) * scaling  # [H, G, n]
    if attention_mask is not None:
        am = attention_mask[0, 0, -1, :n]
        if am.dtype == torch.bool:
            s = s.masked_fill(~am, float("-inf"))
        else:
            s = s + am.float()
    return torch.softmax(s, dim=-1)


def capture_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    out, attn = sdpa_attention_forward(
        module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs
    )
    if STATE.active and query.shape[2] == 1:
        if query.shape[0] != 1:
            raise NotImplementedError("capture expects batch size 1")
        layer = module.layer_idx
        w = _new_token_weights(module, query, key, attention_mask, scaling, kwargs)  # [H, G, n]
        h, g, n = w.shape
        nb = math.ceil(n / BLOCK)
        wb = F.pad(w, (0, nb * BLOCK - n)).view(h, g, nb, BLOCK)
        m = wb.sum(-1)
        q = (wb * wb).sum(-1)
        if not STATE.per_query_head:
            m = m.amax(1)
            q = q.amax(1)
        STATE.m[layer, ..., :nb] = m
        STATE.q[layer, ..., :nb] = q
        if STATE.verify:
            # Check the recomputed weights against the model's own attention output.
            ref = torch.bmm(w.reshape(h, g, n), value[0].float()).reshape(h * g, -1)  # [Hq, D]
            got = out[0, 0].float()  # sdpa returns [B, q_len, Hq, D]
            STATE.verify_err[layer] = ((ref - got).norm() / got.norm().clamp_min(1e-12)).item()
    return out, attn


AttentionInterface.register(ATTN_IMPL, capture_sdpa)
AttentionMaskInterface.register(ATTN_IMPL, sdpa_mask)


# --------------------------------------------------------------------------- helpers


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def code_version() -> dict:
    here = Path(__file__).resolve()
    info = {"script": str(here), "script_sha256": hashlib.sha256(here.read_bytes()).hexdigest()}
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=here.parent, stderr=subprocess.DEVNULL, text=True
        ).strip()
        info["git_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=here.parent, text=True).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["git_commit"] = None
        print("warning: not a git repository; recording the script's sha256 instead of a commit", file=sys.stderr)
    return info


def cached_keys(cache, layer: int) -> torch.Tensor:
    if hasattr(cache, "layers"):
        return cache.layers[layer].keys
    return cache.key_cache[layer]


@torch.inference_mode()
def check_keys_cached_post_rope(model, device, start_pos: int, n_tokens: int = 64) -> dict:
    """Confirm the cache stores keys with RoPE already applied (docs/PHASE0.md, A).

    Runs n_tokens at positions start_pos.. , recomputes layer 0's keys from k_proj,
    applies the model's own rotary embedding and compares both the rotated and the
    unrotated keys with what the cache holds.

    Dims that barely rotate make the two hard to tell apart (Qwen2.5 has large key
    biases in low-frequency dims, so over the whole vector the unrotated keys differ
    by only ~3-6% even at position 16K). So the comparison uses only the dims that
    the model's own cos table shows rotated by more than 60 degrees (cos < 0.5) at
    every test position, and requires a large error against unrotated keys there.
    A late start position makes more dims qualify.
    """
    attn = model.model.layers[0].self_attn
    mod = sys.modules[type(attn).__module__]
    grabbed = {}
    hook = attn.k_proj.register_forward_hook(lambda m, i, o: grabbed.__setitem__("k", o))
    ids = torch.randint(100, 1000, (1, n_tokens), generator=torch.Generator().manual_seed(0)).to(device)
    cache = DynamicCache()
    try:
        pos = torch.arange(start_pos, start_pos + n_tokens, device=device)[None]
        model(input_ids=ids, position_ids=pos, past_key_values=cache, use_cache=True)
    finally:
        hook.remove()
    k = grabbed["k"]
    hd = attn.head_dim
    k_pre = k.view(1, n_tokens, -1, hd).transpose(1, 2)
    if getattr(attn, "k_norm", None) is not None:
        k_pre = attn.k_norm(k_pre)
    cos, sin = model.model.rotary_emb(k_pre, pos)
    _, k_rot = mod.apply_rotary_pos_emb(k_pre, k_pre, cos, sin)
    cached = cached_keys(cache, 0)

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    dims = (cos[0] < 0.5).all(0)  # [head_dim]
    sel = lambda t: t[..., dims]
    res = {
        "start_pos": start_pos,
        "rotated_dims_compared": int(dims.sum()),
        "head_dim": hd,
        "rel_err_vs_rotated": rel(sel(cached), sel(k_rot)),
        "rel_err_vs_unrotated": rel(sel(cached), sel(k_pre)),
        "rel_err_vs_unrotated_all_dims": rel(cached, k_pre),
    }
    res["post_rope"] = (
        res["rotated_dims_compared"] >= hd // 8
        and res["rel_err_vs_rotated"] < 1e-2
        and res["rel_err_vs_unrotated"] > 0.5
    )
    return res


def load_prompts(path: Path, limit: int | None):
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "id" not in rec or not ({"prompt", "messages"} & rec.keys()):
                raise ValueError(f"each line of {path} needs 'id' and 'prompt' or 'messages'")
            prompts.append(rec)
    return prompts[:limit] if limit else prompts


def encode(tok, rec, use_template: bool, max_context: int) -> tuple[list[int], bool]:
    """Token ids for one prompt, middle-truncated to max_context (LongBench convention)."""
    if use_template:
        messages = rec.get("messages") or [{"role": "user", "content": rec["prompt"]}]
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    else:
        ids = tok(rec["prompt"], add_special_tokens=True)["input_ids"]
    truncated = len(ids) > max_context
    if truncated:
        half = max_context // 2
        ids = ids[:half] + ids[len(ids) - (max_context - half):]
    return ids, truncated


def captured_steps(max_new: int, every: int, window: int) -> list[int]:
    return [t for t in range(max_new) if t % every < window]


# --------------------------------------------------------------------------- main loop


@torch.inference_mode()
def run_prompt(model, tok, ids, args, device, n_layers, n_kv, group, eos_ids):
    steps_to_capture = set(captured_steps(args.max_new_tokens, args.capture_every, args.capture_window))
    n0 = len(ids)
    b_max = math.ceil((n0 + args.max_new_tokens) / BLOCK)
    shape_dev = (n_layers, n_kv, group, b_max) if args.per_query_head else (n_layers, n_kv, b_max)
    STATE.m = torch.zeros(shape_dev, dtype=torch.float32, device=device)
    STATE.q = torch.zeros_like(STATE.m)
    STATE.per_query_head = args.per_query_head
    store = np.float16 if args.store_dtype == "float16" else np.float32
    q_is_sqrt = args.store_dtype == "float16"
    rows_m, rows_q, n_ctx, step_idx = [], [], [], []

    input_ids = torch.tensor([ids], device=device)
    cache = DynamicCache()
    STATE.active = False
    sync(device)
    t0 = time.perf_counter()
    for start in range(0, n0, args.prefill_chunk):
        chunk = input_ids[:, start : start + args.prefill_chunk]
        pos = torch.arange(start, start + chunk.shape[1], device=device)
        out = model(input_ids=chunk, past_key_values=cache, use_cache=True, cache_position=pos, logits_to_keep=1)
    sync(device)
    t_prefill = time.perf_counter() - t0

    gen = []
    tok_next = out.logits[:, -1].argmax(-1)
    verify_err = None
    t0 = time.perf_counter()
    for t in range(args.max_new_tokens):
        gen.append(int(tok_next))
        if not args.ignore_eos and int(tok_next) in eos_ids:
            break
        cap = t in steps_to_capture
        STATE.active = cap
        STATE.verify = args.verify and verify_err is None
        if cap:
            STATE.reset_step()
        pos = torch.tensor([n0 + t], device=device)
        out = model(input_ids=tok_next[:, None], past_key_values=cache, use_cache=True, cache_position=pos)
        STATE.active = False
        if cap:
            n = n0 + t + 1
            nb = math.ceil(n / BLOCK)
            rows_m.append(STATE.m.cpu().numpy().astype(store))
            q_store = STATE.q.sqrt() if q_is_sqrt else STATE.q
            rows_q.append(q_store.cpu().numpy().astype(store))
            n_ctx.append(n)
            step_idx.append(t)
            if STATE.verify:
                verify_err = max(STATE.verify_err.values())
                STATE.verify_err.clear()
            # Sanity: per query head the block masses sum to one.
            msum = rows_m[-1][..., :nb].astype(np.float64).sum(-1)
            if args.per_query_head and not np.allclose(msum, 1.0, atol=1e-3):
                raise RuntimeError(f"block masses do not sum to 1 at step {t}: {msum.min()}..{msum.max()}")
            if not args.per_query_head and msum.min() < 1 - 1e-3:
                raise RuntimeError(f"max-reduced block masses sum below 1 at step {t}: {msum.min()}")
        tok_next = out.logits[:, -1].argmax(-1)
    sync(device)
    t_decode = time.perf_counter() - t0

    b_last = math.ceil(n_ctx[-1] / BLOCK) if n_ctx else 0
    arrays = {
        "m": np.stack(rows_m)[..., :b_last] if rows_m else np.zeros((0,) + shape_dev, store),
        "q": np.stack(rows_q)[..., :b_last] if rows_q else np.zeros((0,) + shape_dev, store),
        "n_ctx": np.asarray(n_ctx, np.int64),
        "step": np.asarray(step_idx, np.int64),
        "gen_ids": np.asarray(gen, np.int64),
        "prompt_len": np.int64(n0),
        "block_size": np.int64(BLOCK),
        "q_is_sqrt": np.bool_(q_is_sqrt),
    }
    stats = {
        "prompt_len": n0,
        "generated": len(gen),
        "captured_steps": len(n_ctx),
        "prefill_s": round(t_prefill, 3),
        "decode_s": round(t_decode, 3),
        "decode_tok_per_s": round(len(gen) / t_decode, 2) if t_decode > 0 else None,
        "max_verify_rel_err": verify_err,
    }
    return arrays, stats


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Hugging Face model id or local path")
    p.add_argument("--revision", default=None, help="model revision (branch, tag or commit)")
    p.add_argument("--prompts", type=Path, required=True, help="JSONL with 'id' and 'prompt' (or 'messages')")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--max-context", type=int, required=True, help="prompt tokens kept (middle truncation)")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=None, help="only the first N prompts")
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--chat-template", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--prefill-chunk", type=int, default=2048, help="prefill tokens per forward pass")
    p.add_argument("--capture-every", type=int, default=1, help="capture steps t with t %% every < window")
    p.add_argument("--capture-window", type=int, default=1, help="consecutive steps captured per period")
    p.add_argument("--kv-reduce", default="none", choices=["none", "max"],
                   help="none: keep every query head (default); max: store max over each KV head's query heads")
    p.add_argument("--store-dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--ignore-eos", action="store_true", help="keep decoding past EOS")
    p.add_argument("--verify", action="store_true", help="check recomputed weights against SDPA output once per prompt")
    p.add_argument("--skip-rope-check", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", help="redo prompts already in done.jsonl")
    args = p.parse_args()
    args.per_query_head = args.kv_reduce == "none"
    if args.capture_window < 1 or args.capture_every < args.capture_window:
        p.error("need 1 <= --capture-window <= --capture-every")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=dtype, attn_implementation=ATTN_IMPL
    ).to(device)
    model.eval()
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    group = n_heads // n_kv
    use_template = args.chat_template == "on" or (args.chat_template == "auto" and tok.chat_template is not None)
    eos = cfg.eos_token_id if cfg.eos_token_id is not None else tok.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    if getattr(model, "generation_config", None) is not None and model.generation_config.eos_token_id is not None:
        g_eos = model.generation_config.eos_token_id
        eos_ids |= set(g_eos if isinstance(g_eos, list) else [g_eos])

    rope = None
    if not args.skip_rope_check:
        rope = check_keys_cached_post_rope(model, device, start_pos=max(args.max_context, 4096))
        print(f"RoPE check: {rope}")
        if not rope["post_rope"]:
            sys.exit("cached keys do not look post-RoPE; the design assumption fails for this model (see PHASE0.md A)")

    prompts = load_prompts(args.prompts, args.limit)
    steps = captured_steps(args.max_new_tokens, args.capture_every, args.capture_window)
    bytes_per_val = 4 if args.store_dtype == "float32" else 2
    est = 2 * len(steps) * n_layers * n_kv * (group if args.per_query_head else 1) * math.ceil(
        (args.max_context + args.max_new_tokens) / BLOCK
    ) * bytes_per_val
    print(
        f"device={device} model={args.model} layers={n_layers} q_heads={n_heads} kv_heads={n_kv} "
        f"prompts={len(prompts)} capture_steps<={len(steps)} upper bound per prompt={est / 2**20:.1f} MiB"
    )

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "seed": args.seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or platform.machine(),
        "model": args.model,
        "model_revision": getattr(cfg, "_commit_hash", None) or args.revision,
        "model_shape": {"layers": n_layers, "q_heads": n_heads, "kv_heads": n_kv, "head_dim": getattr(cfg, "head_dim", cfg.hidden_size // n_heads)},
        "block_size": BLOCK,
        "reduction": "none (per query head)" if args.per_query_head else "max over query heads of each KV head",
        "q_stored": "sqrt(q_b)" if args.store_dtype == "float16" else "q_b",
        "chat_template": use_template,
        "eos_ids": sorted(eos_ids),
        "rope_check": rope,
        "code": code_version(),
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__},
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    done_path = args.out / "done.jsonl"
    done = set()
    if done_path.exists() and not args.overwrite:
        done = {json.loads(l)["id"] for l in done_path.read_text().splitlines() if l.strip()}

    for i, rec in enumerate(prompts):
        pid = str(rec["id"])
        if pid in done:
            print(f"[{i + 1}/{len(prompts)}] {pid}: already done, skipping")
            continue
        ids, truncated = encode(tok, rec, use_template, args.max_context)
        arrays, stats = run_prompt(model, tok, ids, args, device, n_layers, n_kv, group, eos_ids)
        fname = args.out / f"{pid}.npz"
        np.savez(fname, **arrays)
        stats.update(
            id=pid,
            truncated=truncated,
            file=fname.name,
            bytes=fname.stat().st_size,
            completion=tok.decode(arrays["gen_ids"], skip_special_tokens=True)[:200],
        )
        with open(done_path, "a") as f:
            f.write(json.dumps(stats) + "\n")
        print(
            f"[{i + 1}/{len(prompts)}] {pid}: prompt={stats['prompt_len']} gen={stats['generated']} "
            f"captured={stats['captured_steps']} prefill={stats['prefill_s']}s decode={stats['decode_s']}s "
            f"file={stats['bytes'] / 2**20:.1f} MiB verify_err={stats['max_verify_rel_err']}"
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
