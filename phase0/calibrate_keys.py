#!/usr/bin/env python
"""Calibrate static key quantiser ranges for tiered.py --key-codecs static.

Prefills a few calibration prompts (kept apart from evaluation prompts) and records,
for every layer, KV head and channel of the cached post-RoPE keys: min, max, mean and
mean absolute deviation. tiered.py uses min/max for min-max quantisers and mean/MAD for
the mean-abs 1-bit plane; out-of-range keys saturate.

Also records how many keys and values fall outside fp8 e4m3's +-448 range (which the
fp8 variant clips), and the largest |key| and |value| per layer.

Output: a torch file {"model", "model_revision", "prompts", "tokens", ..., "layers":
{layer: {"min", "max", "mean", "mad"} each [H, D] float32}}.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import cached_keys, code_version, encode, load_prompts, pick_device  # noqa: E402


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--prompts", type=Path, required=True, help="calibration JSONL (e.g. *.calib.jsonl from benchmarks.py)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=4)
    p.add_argument("--max-context", type=int, default=4096)
    p.add_argument("--prefill-chunk", type=int, default=2048)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16")
    args = p.parse_args()
    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=getattr(torch, args.dtype), attn_implementation="sdpa"
    ).to(device).eval()
    L = model.config.num_hidden_layers
    keys = {l: [] for l in range(L)}
    vmax = torch.zeros(L)
    v_over = torch.zeros(L)
    n_vals = 0
    prompts = load_prompts(args.prompts, args.limit)
    for rec in prompts:
        ids, _ = encode(tok, rec, True, args.max_context)
        cache = DynamicCache()
        x = torch.tensor([ids], device=device)
        for s in range(0, len(ids), args.prefill_chunk):
            model(input_ids=x[:, s : s + args.prefill_chunk], past_key_values=cache, use_cache=True,
                  cache_position=torch.arange(s, min(s + args.prefill_chunk, len(ids)), device=device), logits_to_keep=1)
        for l in range(L):
            keys[l].append(cached_keys(cache, l)[0].float().cpu())  # [H, n, D]
            vals = cache.layers[l].values[0].float().abs() if hasattr(cache, "layers") else cache.value_cache[l][0].float().abs()
            vmax[l] = max(vmax[l], vals.max().item())
            v_over[l] += (vals > 448).sum().item()
        n_vals += len(ids)
        print(f"{rec['id']}: {len(ids)} tokens")
    layers, kmax, k_over = {}, [], []
    for l in range(L):
        k = torch.cat(keys[l], 1)  # [H, N, D]
        mean = k.mean(1)
        layers[l] = {"min": k.amin(1), "max": k.amax(1), "mean": mean, "mad": (k - mean[:, None]).abs().mean(1)}
        kmax.append(k.abs().max().item())
        k_over.append((k.abs() > 448).float().mean().item())
    H, D = layers[0]["min"].shape
    blob = {
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None) or args.revision,
        "prompts": [r["id"] for r in prompts],
        "tokens": n_vals,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code": code_version(),
        "max_abs_key_per_layer": kmax,
        "max_abs_value_per_layer": vmax.tolist(),
        "frac_keys_beyond_fp8_range_per_layer": k_over,
        "frac_values_beyond_fp8_range_per_layer": (v_over / (n_vals * H * D)).tolist(),
        "layers": layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    print(f"{len(prompts)} prompts, {n_vals} tokens -> {args.out}")
    print(f"max |key| {max(kmax):.1f} (layer {kmax.index(max(kmax))}), keys beyond fp8 range: {max(k_over):.2e} of the worst layer; "
          f"max |value| {vmax.max().item():.1f}")


if __name__ == "__main__":
    main()
