# Results

Small analysis outputs copied from `runs/` (which is not tracked). The large capture `.npz`
files stay in `runs/` and are ignored.

`prescreen-4k/<model>/`: 40 LongBench continuation prompts at 4K tokens, 128 teacher-forced
reference tokens each, MPS. Produced by `phase0/analyze_d.py --runs runs/prescreen-4k/<model>`.

- `summary.csv`: one row per variant (perplexity increase with 95% bootstrap CI, rho, tiers,
  fix-ups, COLD-mass tail, worst prompt and prompts above +2%).
- `decision.json`: the frozen go rule of `docs/PHASE0.md` applied to `summary.csv`; its
  `comparison` block puts the bar and uniform 4-bit (group256) beside variants e and f.
- `by_length.csv`: HOT/WARM/COLD fractions by context length.
- `meta.json`: analysis config, git commit, and each run's config, seed and model revision
  (paths made relative to the repo).
- `NOTES.md`: run notes, where there are any.
