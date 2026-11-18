#!/usr/bin/env python
"""Capture for docs/JUMPS.md: variant e's stale plan and the attention it misses, per head and block.

Runs variant e (coldkeys_R4, eps_w 1e-3, eps_c 0.02, ladder 4/3, group256 key codec unless
overridden) teacher-forced on each record's reference, exactly as tiered.py does, and records
per decode step:
    plan    [T, L, H, NB] int8   the stale plan before promotion (tiered.py `planned`);
                                 PAD (9) for blocks that do not exist yet at that step
    cold    [T, L, Hq] float32   true attention mass on planned-COLD blocks, per query head
    jump_*                       for every (step, layer, query head) with cold mass > JUMP:
                                 jump_idx [J, 3] (t, layer, q head), jump_mass [J, NB] float16
                                 (true mass per block, zero outside planned-COLD blocks)
    nll, tail                    as tiered.py, to check the run against the committed results
plus prompt.json with the context ids (prompt with BOS, then the reference).

--repeat N replaces the prompt file by N synthetic positive-control records (docs/JUMPS.md):
2048 random token ids repeated twice, prompt = first 3967 tokens, reference = the last 129.

  python phase0/jumps_capture.py --model Qwen/Qwen2.5-1.5B-Instruct --prompts data/prompts/qwen_cont_4k.jsonl \
      --out runs/jumps/qwen2.5-1.5b
  python phase0/jumps_capture.py --model ... --repeat 4 --out runs/jumps/qwen2.5-1.5b-repeat
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiered  # noqa: E402
from analyze import COLD  # noqa: E402
from capture import code_version, encode, load_prompts, pick_device, sync, vocab_sha  # noqa: E402

JUMP = 0.05
PAD = 9
REPEAT_LEN, REPEAT_PROMPT, REPEAT_REF = 2048, 3967, 129


def repeat_records(tok, n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        s = rng.integers(1000, 30000, REPEAT_LEN).tolist()
        full = s + s
        out.append({"id": f"repeat-{seed}-{i}", "prompt_ids": full[:REPEAT_PROMPT],
                    "reference_ids": full[REPEAT_PROMPT:REPEAT_PROMPT + REPEAT_REF],
                    "tokenizer_vocab_sha": vocab_sha(tok), "n_tokens": REPEAT_PROMPT})
    return out


class Recorder:
    """tiered.STATE.dump target: keeps one step's per-layer plan and COLD mass on the device."""

    def __init__(self, n_layers: int):
        self.L = n_layers
        self.layers = {}
        self.steps = []

    def __call__(self, layer, planned, m16):
        cold = (planned == COLD)[:, None, :]  # [H, 1, nb]
        mc = m16 * cold  # [H, G, nb]
        self.layers[layer] = (planned.clone(), mc.reshape(-1, mc.shape[-1]).half(), mc.sum(-1).reshape(-1))

    def end_step(self):
        plan = torch.stack([self.layers[l][0] for l in range(self.L)])  # [L, H, nb]
        mass = torch.stack([self.layers[l][1] for l in range(self.L)])  # [L, Hq, nb]
        cold = torch.stack([self.layers[l][2] for l in range(self.L)])  # [L, Hq]
        sel = cold > JUMP
        idx = sel.nonzero()
        self.steps.append((plan.cpu().numpy(), cold.cpu().numpy(), idx.cpu().numpy(), mass[sel].cpu().numpy()))
        self.layers = {}

    def arrays(self) -> dict:
        T = len(self.steps)
        L, H = self.steps[0][0].shape[:2]
        nbmax = max(s[0].shape[-1] for s in self.steps)
        plan = np.full((T, L, H, nbmax), PAD, np.int8)
        jidx, jmass = [], []
        for t, (p, c, idx, m) in enumerate(self.steps):
            plan[t, :, :, : p.shape[-1]] = p
            if len(idx):
                jidx.append(np.concatenate([np.full((len(idx), 1), t), idx], 1))
                jm = np.zeros((len(idx), nbmax), np.float16)
                jm[:, : m.shape[-1]] = m
                jmass.append(jm)
        return {"plan": plan, "cold": np.stack([s[1] for s in self.steps]).astype(np.float32),
                "jump_idx": np.concatenate(jidx).astype(np.int32) if jidx else np.zeros((0, 3), np.int32),
                "jump_mass": np.concatenate(jmass) if jmass else np.zeros((0, nbmax), np.float16)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--prompts", type=Path, default=None)
    p.add_argument("--repeat", type=int, default=0, help="N synthetic repeated-random-token prompts instead of --prompts")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-context", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=129)
    p.add_argument("--warm-bits", type=int, default=3)
    p.add_argument("--hot-bits", type=int, default=4)
    p.add_argument("--key-codec", default="group256")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--chat-template", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--prefill-chunk", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if (args.prompts is None) == (args.repeat == 0):
        sys.exit("give --prompts or --repeat N")

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    variant = tiered.Variant("coldkeys", bits=args.warm_bits, R=4, eps_w=1e-3, eps_c=0.02, kcodec=args.key_codec,
                             hot_bits=args.hot_bits)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.float16,
                                                 attn_implementation=tiered.ATTN_IMPL).to(device).eval()
    cfg = model.config
    tiered.STATE.L = cfg.num_hidden_layers
    use_template = args.chat_template == "on" or (args.chat_template == "auto" and tok.chat_template is not None)
    if args.repeat:
        prompts = repeat_records(tok, args.repeat, args.seed)
        with open(args.out / "repeat_prompts.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in prompts)
    else:
        prompts = load_prompts(args.prompts, args.limit)
    jumps_doc = Path(__file__).resolve().parent.parent / "docs" / "JUMPS.md"
    import hashlib
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "variants": [variant.__dict__ | {"name": variant.name}],
        "jump_threshold": JUMP, "seed": args.seed, "device": str(device),
        "model": args.model, "model_revision": getattr(cfg, "_commit_hash", None) or args.revision,
        "model_shape": {"layers": cfg.num_hidden_layers, "q_heads": cfg.num_attention_heads,
                        "kv_heads": cfg.num_key_value_heads},
        "block_size": tiered.BLOCK, "code": code_version(), "versions": tiered.versions(),
        "prereg_sha256": hashlib.sha256(jumps_doc.read_bytes()).hexdigest(),
    }
    meta_path = args.out / "meta.json"
    tiered.check_versions(meta_path, meta["versions"], False)
    tiered.write_meta(meta_path, meta)
    done_path = args.out / "done.jsonl"
    done = {json.loads(l)["id"] for l in done_path.read_text().splitlines() if l.strip()} if done_path.exists() else set()
    print(f"device={device} variant={variant.name} prompts={len(prompts)} done={len(done)}")

    for i, rec in enumerate(prompts):
        pid = str(rec["id"])
        if pid in done:
            continue
        ids, truncated = encode(tok, rec, use_template, args.max_context)
        n0 = len(ids)
        teacher = list(rec["reference_ids"])[: args.max_new_tokens]
        cache = DynamicCache()
        tiered.STATE.active = False
        t0 = time.perf_counter()
        x = torch.tensor([ids], device=device)
        with torch.inference_mode():
            for s in range(0, n0, args.prefill_chunk):
                c = x[:, s : s + args.prefill_chunk]
                model(input_ids=c, past_key_values=cache, use_cache=True,
                      cache_position=torch.arange(s, s + c.shape[1], device=device), logits_to_keep=1)
        rec_ = Recorder(cfg.num_hidden_layers)
        tiered.STATE.variant, tiered.STATE.dump = variant, rec_
        orig_end = tiered.TieredState.end_step

        def end_step(self=tiered.STATE):
            rec_.end_step()
            return orig_end(self)

        tiered.STATE.end_step = end_step
        try:
            arr = tiered.decode(model, cache, n0, teacher[0], device, teacher=teacher)
        finally:
            tiered.STATE.dump = None
            del tiered.STATE.end_step
        sync(device)
        secs = time.perf_counter() - t0
        out = rec_.arrays() | {"nll": arr["nll"], "tail": arr["tail"]}
        pdir = args.out / pid
        pdir.mkdir(exist_ok=True)
        np.savez_compressed(pdir / "jumps.npz", **out)
        (pdir / "prompt.json").write_text(json.dumps({"id": pid, "prompt_len": n0, "truncated": truncated,
                                                      "context_ids": ids + teacher}))
        with open(done_path, "a") as f:
            f.write(json.dumps({"id": pid, "prompt_len": n0, "steps": len(arr["nll"]), "jumps": int(len(out["jump_idx"])),
                                "seconds": round(secs, 1)}) + "\n")
        print(f"[{i + 1}/{len(prompts)}] {pid}: ppl={math.exp(arr['nll'].mean()):.4f} jumps={len(out['jump_idx'])} "
              f"({secs:.0f}s)", flush=True)
        del cache
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
