#!/usr/bin/env python
"""Repro for a PyTorch issue: KV-cache decode on MPS intermittently gives wrong results
(NaN in fp16, wrong but finite in fp32), depending on what runs between kernels.

Model: ibm-granite/granite-3.3-2b-instruct (Apache-2.0, ungated, ~5 GB), MPS,
attn_implementation="sdpa". A 4096-token prompt is prefilled into a DynamicCache in
2048-token chunks, then 128 tokens are decoded one at a time (teacher-forced) and the
next-token NLL is recorded. A correct decode is deterministic, so any run whose
perplexity differs from the others (beyond fp16-vs-fp32 rounding) is a failure.

Modes (each repeated --runs times in one process):

  plain   fp16, no instrumentation.
  check   fp16; torch.nn.functional.scaled_dot_product_attention is wrapped and, after
          each decode call (q_len == 1), the wrapper reads `torch.isfinite(out).all()`
          into Python. Read-only: it changes no tensor.
  sync    fp16; the wrapper calls torch.mps.synchronize() after every SDPA call instead.
  eager   fp16, attn_implementation="eager", no instrumentation.
  fp32    plain, model in float32.

Observed on an M2 Pro (32 GB), macOS 26.6.2, torch 2.9.0, transformers 4.57.3, on one
LongBench gov_report prompt (--prompts ... --id gov_report-6de6); correct ppl is 1.4585
(fp16) / 1.4584 (fp32):

  check   NaN in 3 of 4 runs (first non-finite at decode step 2). At the first bad SDPA
          call (layer 1) the new token's q and k already hold inf/NaN next to finite
          values near 6.3e4; v is finite.
  plain   NaN in 2 of 6 runs (fresh processes), correct in 4
  sync    correct in 2 of 2
  eager   correct in 2 of 2 (1.4578)
  fp32    correct in run 0; run 1 (later in the same process) gave ppl 4.0842, finite

Re-run 2026-11-17 with torch 2.15.0 (latest release; transformers 4.57.3, same macOS and
machine), 3 runs per mode on the same prompt: no failures in 15 runs (check, plain, sync
1.4588; eager 1.4589; fp32 1.4584). Under 2.9.0 the check mode failed in 3 of 4 runs, so a
fix between 2.9.0 and 2.15.0 is likely; check the 2.10-2.15 release notes before filing.

With forward hooks on every module, a correct decode's largest activation anywhere is ~314,
far below the fp16 range, so the NaNs are not an overflow of real values. Read-only
instrumentation changes the outcome, and fp32 also fails once: this looks like a race or
buffer-reuse problem in the MPS backend. Not yet reduced below a full model. On a synthetic
random-token prompt (the default), check mode failed in 1 of 2 runs.

Usage:
  python tools/mps_decode_repro.py                       # synthetic prompt, all modes
  python tools/mps_decode_repro.py --modes check plain --runs 3
  python tools/mps_decode_repro.py --prompts data/prompts/granite_cont_4k.jsonl --id gov_report-6de6

Exit status 1 if any run produced a non-finite NLL (wrong-but-finite runs show up only as
a different perplexity).
"""
import argparse
import json
import math
import platform
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, DynamicCache

MODEL = "ibm-granite/granite-3.3-2b-instruct"
REVISION = "707f574c62054322f6b5b04b6d075f0a8f05e0f0"


def synthetic_prompt(n_prompt, n_ref, vocab, seed):
    """A fixed pseudo-random token sequence: no external data needed."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, vocab - 1000, (n_prompt + n_ref,), generator=g).tolist()
    return ids[:n_prompt], ids[n_prompt:]


def load_prompt(args, vocab):
    if not args.prompts:
        return f"synthetic(seed={args.seed})", *synthetic_prompt(args.n_prompt, args.n_ref, vocab, args.seed)
    for line in open(args.prompts):
        rec = json.loads(line)
        if args.id in rec["id"]:
            return rec["id"], rec["prompt_ids"], rec["reference_ids"]
    sys.exit(f"no record matching {args.id!r} in {args.prompts}")


def run(mode, ids, ref, chunk):
    dev = torch.device("mps")
    dtype = torch.float32 if mode == "fp32" else torch.float16
    impl = "eager" if mode == "eager" else "sdpa"
    sdpa = torch.nn.functional.scaled_dot_product_attention
    first_bad = {}

    def wrapped(q, k, v, *a, **kw):
        out = sdpa(q, k, v, *a, **kw)
        if mode == "sync":
            torch.mps.synchronize()
        elif mode == "check" and q.shape[2] == 1 and not first_bad:
            if not bool(torch.isfinite(out).all()):  # read-only check
                first_bad.update(q_finite=bool(torch.isfinite(q).all()), k_finite=bool(torch.isfinite(k).all()),
                                 v_finite=bool(torch.isfinite(v).all()))
        return out

    torch.nn.functional.scaled_dot_product_attention = wrapped if mode in ("check", "sync") else sdpa
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=dtype,
                                                     attn_implementation=impl).to(dev).eval()
        cache, x, nll = DynamicCache(), torch.tensor([ids], device=dev), []
        with torch.inference_mode():
            for s in range(0, len(ids), chunk):
                e = min(s + chunk, len(ids))
                model(input_ids=x[:, s:e], past_key_values=cache, use_cache=True,
                      cache_position=torch.arange(s, e, device=dev), logits_to_keep=1)
            for t in range(len(ref) - 1):
                out = model(input_ids=torch.tensor([[ref[t]]], device=dev), past_key_values=cache, use_cache=True,
                            cache_position=torch.tensor([len(ids) + t], device=dev))
                nll.append(-torch.log_softmax(out.logits[0, -1].float(), -1)[ref[t + 1]].item())
    finally:
        torch.nn.functional.scaled_dot_product_attention = sdpa
        del model
        torch.mps.empty_cache()
    bad = [i for i, v in enumerate(nll) if not math.isfinite(v)]
    ppl = math.exp(sum(nll) / len(nll)) if not bad else float("nan")
    return ppl, bad, first_bad


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--modes", nargs="+", default=["check", "plain", "sync", "eager", "fp32"],
                   choices=["plain", "check", "sync", "eager", "fp32"])
    p.add_argument("--runs", type=int, default=2, help="repeats per mode (the plain-mode failure is intermittent)")
    p.add_argument("--prompts", help="jsonl with id, prompt_ids, reference_ids (default: synthetic tokens)")
    p.add_argument("--id", default="gov_report-6de6", help="substring of the record id to use with --prompts")
    p.add_argument("--n-prompt", type=int, default=4096)
    p.add_argument("--n-ref", type=int, default=129)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chunk", type=int, default=2048)
    args = p.parse_args()
    if not torch.backends.mps.is_available():
        sys.exit("needs an MPS device")
    print(f"torch {torch.__version__}  transformers {transformers.__version__}  macOS {platform.mac_ver()[0]}  "
          f"{platform.machine()}  python {platform.python_version()}")
    vocab = transformers.AutoConfig.from_pretrained(MODEL, revision=REVISION).vocab_size
    name, ids, ref = load_prompt(args, vocab)
    print(f"model {MODEL}@{REVISION[:7]}  prompt {name}: {len(ids)} prompt + {len(ref) - 1} decode steps")
    failed = False
    for mode in args.modes:
        for r in range(args.runs):
            ppl, bad, info = run(mode, ids, ref, args.chunk)
            failed |= bool(bad)
            print(f"  {mode:6} run {r}: ppl {ppl:.4f}  non-finite steps {len(bad)}"
                  + (f" (first {bad[0]})" if bad else "") + (f"  at first bad SDPA call: {info}" if info else ""), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
