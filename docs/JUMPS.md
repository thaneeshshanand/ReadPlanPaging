# Can token matches predict the jumps a stale plan misses? (side analysis, no GPU)

Written 2026-11-18, before any capture or result of this analysis existed. Everything below,
including the implementation choices that the request left open, is fixed here and is not
changed after results are seen.

## Question

Variant e's stale plan (R = 4) sometimes misses attention that moves onto blocks it marked
COLD. Can (1) a short memory of recently non-COLD blocks and (2) a lexical lookup on the
current input token find those blocks, better than random picks of the same size?

## Data

- Models and prompts: the three 4K pre-screen models and their committed prompt files, at the
  committed revisions. Qwen2.5-1.5B and Qwen2.5-3B on `data/prompts/qwen_cont_4k.jsonl`,
  Granite 3.3 2B on `data/prompts/granite_cont_4k.jsonl` (40 prompts each, 4096 tokens, 128
  scored teacher-forced steps over the record's reference).
- The committed `.npz` results keep only per-layer totals, so per-head, per-block attention is
  recaptured on the laptop by `phase0/jumps_capture.py` (capture) and read by
  `phase0/jumps_analyze.py` (analysis).
- Variant e exactly as committed: `coldkeys_R4`, eps_w 1e-3, eps_c 0.02, ladder 4/3, key codec
  group256, teacher-forced on the record's reference. Attention weights are the exact fp32
  softmax of the variant's own query against the fp16 cache (the same `m16` that the committed
  `tail` field uses), so the trajectory is variant e's own.

## Definitions

- Plan: variant e's stale plan before same-step promotion (`planned` in tiered.py): rule B on
  the exact weights every 4 steps, max tier over the query heads of each KV head, blocks
  created since the refresh HOT.
- Jump: a (query head, step) whose true attention mass on blocks its KV head's plan marks COLD
  exceeds 0.05. Jump mass: that mass. (Per query head, as the committed `tail` counts; the
  blocks are chosen per KV head, because the cache and the plan are per KV head.)
- Cap: 5% of blocks = floor(0.05 x current block count), per KV head and step (13 at 4K).
- Step 1, memory: COLD blocks that were not COLD in the plan at any of the previous 32 steps
  (t-32 .. t-1; a block that did not exist yet counts as not COLD, since new blocks are HOT),
  ordered by the most recent such step, then by block index, newest first. Up to the cap.
  Leftover = jump mass minus the mass memory covers.
- Step 2 picks up to another cap of blocks among the remaining COLD blocks (COLD, not chosen by
  memory); k2 = min(cap, number of remaining COLD blocks).
  - (a) random: k2 blocks uniformly from the remaining COLD blocks. Reported as the exact
    expectation, k2 / |remaining COLD| x leftover mass, so it has no sampling noise.
  - (b) lexical: for every earlier position p of the current input token (positions before the
    current one, over prompt plus the reference tokens already fed), the block of p and then
    the block of p + 1. Occurrences are ranked by the longest matching suffix (the tokens
    before p against the tokens before the current position: 4, 3, 2, then 1 token), then by
    rarity of the matched tokens (sum of -log of their counts in the prompt), then most
    recent first. Blocks in that order, duplicates dropped, then filtered to the remaining COLD
    blocks of the head; the first k2 are taken.
  - Two matching modes: exact token ids; token text lowercased with leading spaces stripped.
- Coverage shares are pooled: sum of covered mass over all jumps divided by the sum of their
  leftover mass.
- Lookup size: per step, the number of distinct blocks the lexical lookup returns before the
  cap (all tiers), and per head-step the number of those in the remaining COLD set.

## Controls (run first)

- Positive: per model, 4 prompts of 2048 uniformly random token ids (seeded, from ids
  1000 .. 30000), repeated twice; the prompt is the first 3967 tokens and the 129 reference
  tokens are the end of the second repeat, so every scored step is inside it. Heads that fire
  (amendment 2): KV heads whose jumps are sharp, i.e. mass-weighted over the head's jumps, the
  largest single block holds at least 50% of the jump mass. Pass: exact lexical covers at least
  90% of the leftover jump mass in those heads. On failure, check the
  token alignment (block of p versus p + 1) before anything else.
- Negative: the lexical index (positions and their blocks) is built from the next prompt in the
  file (cyclic), truncated to the current context length; the query token and its suffix stay
  the current prompt's. Pass: its pooled coverage is between 0.5 and 2 times that of random
  picks of the same number of blocks the lookup picked (amendment 1).
- If a control fails, the main result is not reported as a verdict until the cause is found.

## Concentration and the scan baseline

- Split: prompts at even positions in the prompt file (0, 2, ...) are the ranking half A; odd
  positions are the held-out half B.
- Rank KV heads (layer, KV head) by total jump mass on half A. On half B, report the share of
  jump mass held by the top 5%, 10% and 20% of heads, k = max(1, round(p x number of KV heads)).
- "Scan the jumpiest heads": the top 10% of heads (from half A) scan all keys every step and
  are taken to miss nothing; all other heads use the stale plan plus memory. Report the leftover
  jump mass on half B, as a share of half B's total jump mass.

## Rule (replaces the earlier rule; fixed before any results)

1. NO-GO (concentrated) if, on at least two of three models, the top 10% of heads hold at
   least 90% of jump mass on the held-out half.
2. Otherwise GO if, on at least two of three models and on the heads outside that top 10%,
   lexical covers at least 40% of the leftover jump mass and at least 5 times what random
   covers.
3. Otherwise NO-GO.

Choices the request left open, fixed here:
- Rule 2 is evaluated on held-out half B, for the heads outside the top 10% ranked on half A.
- Rule 2 uses the exact-token-id lexical lookup (the first version listed). The normalised-text
  version is reported beside it and does not decide; if it would give a different verdict, that
  is said.
- Per-model, per-layer and per-head tables are reported on all 40 prompts and do not decide.

## Amendments

1. 2026-11-18, after a 3-prompt smoke run of Qwen2.5-1.5B (`runs/jumps-smoke`, a pipeline check,
   not a result) and before any full capture. The negative control first compared the
   foreign-index lookup with random picks of the full k2 blocks. A lookup often returns fewer
   than k2 blocks in the remaining COLD set (all the more so with a foreign index), so it would
   fall below full-budget random even with no positional bias, which is not what the control is
   for. The control now compares with random picks of as many blocks as the lookup picked
   (exact expectation, as for (a)). The decision rule is unchanged: it still compares lexical
   with random at the full k2, which can only make lexical look worse, not better. The
   matched-count comparison is also reported for the exact and normalised lookups, and does not
   decide.

2. 2026-11-18, after the Qwen2.5-1.5B positive control (4 prompts) and before any full capture.
   With "heads that fire" defined as the heads holding 80% of the leftover jump mass, exact
   lexical covered 18.6% (FAIL). Alignment was checked first, with separate code: where the
   missed mass is concentrated it sits on the block of p + 1 (offset 0, no +-1 shift), but only
   about 13% of the control's jump mass is there; the rest is spread thinly over many COLD
   blocks, and those diffuse heads dominate the 80%-of-mass set. That definition selects heads
   that are not copying, so it does not test what the control is for. Heads that fire are now
   the heads with sharp jumps (largest block >= 50% of jump mass, mass-weighted); the definition
   does not refer to token positions or to the lookup. The original-definition result is still
   reported. The decision rule is unchanged.

## Control results so far (2026-11-18, laptop, torch 2.15.0, MPS)

Positive control (4 repeat prompts per model; `runs/jumps/*-repeat`, analysis in
`runs/jumps-smoke/analysis/decision.json`). Fails on all three models, under both definitions:

| Model | Firing heads (sharp) | Their share of jump mass | Exact lexical, firing heads | Random | Original top-80% definition |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 2 / 56 | 29.4% | 42.4% | 13.2% | 18.6% |
| Qwen2.5-3B | 5 / 72 | 8.8% | 12.6% | 16.0% | 1.3% |
| Granite 3.3 2B | 4 / 320 | 4.0% | 8.4% | 5.4% | 4.9% |

Diagnosis (scratch script, separate from the analysis code): no token-alignment error. In
Qwen2.5-1.5B's main induction head (layer 6, KV head 1), 43% of its jump mass has its largest
block exactly on the block of p + 1 while the plan had that block COLD, and lexical recovers
about that much. In the remainder, the copy target was already HOT and the >5% on COLD blocks is
thin leakage spread over many unrelated blocks. Qwen2.5-3B is similar (36% on the COLD copy
target). In Granite's sharp heads the largest COLD block sits near the start of the sequence or
just behind the current position, not on the copy target. Since the plan refreshes from exact
weights every 4 steps, it mostly reads the copy target already, so in this control, jumps are
mostly not copy events.

Negative control (3-prompt Qwen2.5-1.5B smoke only): 1.6% against 1.8% for random with as many
blocks (x0.86), pass. Full runs not yet started.

## Additions (2026-11-18, after the positive control, before any full-run data)

The positive control stays recorded as FAILED under both definitions and is not changed
again; the verdict carries that flag. Three numbers are added. They are reported beside the
rule's verdict, per model, and do not change the rule.

a. Ceiling: per jump, the share of its leftover mass held by the remaining COLD blocks with
   the most mass, up to k2 blocks (the same budget as random and lexical). The best any pick of
   that size could do for that jump (an oracle per query head, so it can exceed what a per-KV-head
   pick could reach). Pooled like the others.
b. Sharpness: the share of jumps, and of jump mass, where the largest COLD block holds at least
   50% of the jump mass.
c. Position: the share of jump mass in the first 4 blocks, and in the 8 blocks just behind the
   always-read recent blocks (the last two blocks at the plan's refresh step, and every newer
   block, are HOT).

Alignment check, now a test (`tests/test_jumps.py`): on the repeated random-token prompts
(synthetic, and the captured controls when present), the lookup ranks the block holding the
token after the earlier copy among its first two blocks on every scored step.
