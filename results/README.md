# Results

Small analysis outputs copied from `runs/` (which is not tracked). The large capture `.npz`
files stay in `runs/` and are ignored.

`prescreen-4k/<model>/`: 40 LongBench continuation prompts at 4K tokens, 128 teacher-forced
reference tokens each, MPS. Produced by `phase0/analyze_d.py --runs runs/prescreen-4k/<model>`.

- `summary.csv`: one row per variant (perplexity increase with 95% bootstrap CI, rho, tiers,
  fix-ups, COLD-mass tail, worst and best prompt, prompts above +2%, flag counts). Worst prompt,
  best prompt and prompts above +2% are indicative only: they rest on 128 scored tokens per prompt
  and never decide (`indicative_only` in `decision.json`).
- `decision.json`: the frozen go rule of `docs/PHASE0.md` applied to `summary.csv`; its
  `comparison` block puts the bar and uniform 4-bit (group256) beside variants e and f.
- `per_prompt.csv`: per prompt and variant NLL sums and paired increase over fp16 (the
  reference for `phase0/compare_runs.py --check`, e.g. the GPU `replicate4k` stage).
- `flags.csv`: per-prompt rows with a non-finite value or |perplexity increase| > 10%.
- `by_length.csv`: HOT/WARM/COLD fractions by context length.
- `meta.json`: analysis config, git commit, and each run's config, seed and model revision
  (paths made relative to the repo).
- `NOTES.md`: run notes, where there are any.

## jumps-4k

Side analysis of docs/JUMPS.md (can token matches predict the attention a stale plan misses?):
decision.json (verdict, controls, every reported number), per model summary.json, per_head.csv,
per_layer.csv and the capture meta. Verdict: no-go; the positive control failed (flagged).
