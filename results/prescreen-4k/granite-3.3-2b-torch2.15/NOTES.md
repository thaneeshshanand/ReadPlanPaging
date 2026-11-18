# Granite 3.3 2B, 4K set, rerun under torch 2.15.0 (MPS)

First run in the torch 2.15.0 environment (`requirements.txt`; venv on the laptop), via
`scripts/phase0_run.sh replicate4k` with DEVICE=mps: fp16, K4/V3 (bar), uniform 4-bit, e and
f on 4/3, group256 keys, 40 prompts, same model revision (707f574) and arguments as the
torch 2.9.0 run in `../granite-3.3-2b`. Compared with `phase0/compare_runs.py --flag-pp 0.3`
(`compare.log`).

- Verdict unchanged (go): bar K4/V3 +0.29% (was +0.30%), e +0.11% (+0.19%), f +0.27% (+0.40%);
  rho_meta identical. Pooled changes moved by at most 0.13 points.
- fp16 per-prompt perplexity moved by a median 0.03%.
- Per-prompt paired increases of the quantised variants moved by a median 0.25-0.35 points
  (max 1.85); 81 of 200 rows exceed 0.3 points, on 36 of 40 prompts; 14 rows cross +2%.
- --verify: stock SDPA matched the patched path on all 40 prompts (max ppl ratio diff 0.12%,
  attention rel err <= 6.1e-4); under 2.9.0 one prompt had a broken stock decode.

A one-prompt rerun under 2.9.0 reproduced the committed numbers exactly, so the drift comes
from the torch version, not run-to-run noise. Likely mechanism (not verified): small fp16
differences in keys and values move values across quantiser rounding boundaries, and with
128 scored tokens per prompt a few changed tokens move a prompt's perplexity by ~1%. Treat
per-prompt differences below ~1 point across environments as noise; the pooled numbers are
the stable quantity.

## Re-checked under the replication rule (2026-11-17)

`compare.log`, `compare_*.csv` and `compare_check.json` were regenerated with
`compare_runs.py --check` (the rule of `docs/PHASE0.md`, "Changes made before any 7B data").
The 0.3-point flags above are superseded. Result: PASS. Verdict unchanged; largest pooled
difference -0.12 points (f), limit 0.25; largest per-prompt difference 1.85 points, limit 5.
Per-prompt |difference| by band over the four quantised variants (160 rows): 79 below 0.3,
75 at 0.3-1, 6 at 1-2, none above 2.
