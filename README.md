# Tiered KV-cache reads for LLM decoding

A measurement study of how many KV-cache bytes a language model has to read per generated token, and whether reading the cache at mixed precision on every step beats a simple compressed cache.

**Status:** paused in November 2026 after a 4K-context study on three small models. This is an independent test of a published idea, plus two negative results. It is not a new method, and no serving speed was measured.

## Why this question

When a GPU generates text, attention is limited by memory bandwidth, not compute. Every decode step re-reads the whole KV cache, so at long contexts the bytes read per token set the throughput. vLLM's PagedAttention fixed how that memory is allocated. It still reads every block in full on every step.

This repo measures how far the bytes read can fall at equal output quality.

## Findings

1. **A plain 4-bit cache is the bar to beat.** It reads about a quarter of the full-precision bytes and stays within 1% perplexity. 3-bit fails on every model.
2. **Tiered reads save about a third more.** Read every key at 3 bits on each step, then read each 16-token block's values at 4 bits, 3 bits or not at all. That uses 30-34% fewer bytes than the best simple cache at equal quality.
3. **Stored read plans add nothing.** Deciding fresh on every step matches a plan carried across steps.
4. **Cheap predictors cannot replace the key scan.** What a stale plan misses is spread over many heads and several blocks. Token matching finds it no better than random picks.

| Model | Best simple cache | Tiered read (variant f) | Fewer bytes | Perplexity change, tiered |
| --- | --- | --- | --- | --- |
| Qwen2.5-1.5B | 0.264 | 0.176 | 33% | +0.47% [0.16, 0.74] |
| Qwen2.5-3B | 0.260 | 0.181 | 30% | +0.63% [0.22, 1.03] |
| Granite 3.3 2B | 0.241 | 0.161 | 33% | +0.40% [0.00, 0.81] |

Bytes are a fraction of what a full-precision (fp16) cache reads per step, with quantiser metadata included. Perplexity change is against fp16, with a bootstrap 95% interval over 40 prompts.

## What this is not

- **Not a new method.** The per-step loop is the one published as [ReadKV](https://arxiv.org/abs/2610.11245) in October 2026. This repo tests the idea in a simpler block-level form. It does not reproduce ReadKV's method, and the two sets of numbers are not comparable.
- **Not a speed measurement.** Bytes are counted in simulation. No kernel was written and nothing was timed.
- **Not at scale.** 4K contexts, models of 1.5B to 3B parameters, one laptop.

## Method

- **Models.** Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct and IBM Granite 3.3 2B Instruct, all in fp16.
- **Data.** 40 LongBench documents cut at 4,096 tokens. The next 129 tokens are scored with teacher forcing.
- **Quality.** Perplexity change against fp16.
- **Cost.** Fraction of fp16 KV bytes read per decode step.
- **Blocks.** 16 tokens, as in vLLM.
- **Tiering rule.** Per head, blocks are promoted to the top tier by squared attention weight until the remainder is under a budget (1e-3). The lowest-mass blocks that together hold under 2% of attention are skipped.
- **Decision rule.** Fixed before the deciding runs: pass if a tiered variant stays within 1% perplexity and reads at least 25% fewer bytes than the cheapest uniform cache within 1%.

| Variant | What it does |
| --- | --- |
| a | Fresh plan from exact attention on every step (an oracle) |
| b | Stale plan, no check |
| c | Stale plan plus same-step promotion from a 1-bit key scan |
| d | Two tiers, no skipped blocks |
| e | Stale plan; keys of skipped blocks are still read every step |
| f | No stored plan: read all keys at 3 bits each step, apply the rule, read values by tier |

## Results

### Simple compressed caches

| Cache | Bytes read | Qwen 1.5B | Qwen 3B | Granite 2B |
| --- | --- | --- | --- | --- |
| Uniform 8-bit | 0.51-0.57 | about 0% | about 0% | +0.02% |
| Uniform 4-bit | 0.26-0.27 | +0.55% | +0.61% | +0.20% |
| Keys 4-bit, values 3-bit | 0.23-0.24 | +1.59% | +1.40% | +0.30% |
| Uniform 3-bit | 0.20-0.21 | +3.38% | +1.82% | +1.36% |

Key scales are shared over 256 tokens. Per-block scales cost more (0.32-0.33 at 4 bits) because of metadata.

### Tiered reads

Variants e and f pass the decision rule on all three models with 4-bit and 3-bit tiers. The Qwen 3B passes are marginal: the interval tops are 1.06% and 1.03%.

A more aggressive ladder (4-bit and 2-bit tiers) reads about half the bytes of the bar but breaks the 1% limit on 3 of 4 runs.

### Negative result: predicting what to read

Could a cheap predictor replace the per-step key scan? Under a stale plan, a "jump" is a head-step with more than 5% of attention on blocks the plan skipped. The test asked whether token matches (blocks where the current input token appeared before) find that missed attention.

| Model | Top 10% of heads hold | Token matching | Random picks | Best possible 13 blocks |
| --- | --- | --- | --- | --- |
| Qwen 1.5B | 32.3% | 14.1% | 14.8% | 60.7% |
| Qwen 3B | 33.1% | 14.0% | 15.3% | 62.7% |
| Granite 2B | 35.2% | 10.9% | 9.7% | 56.2% |

- The misses are not in a few heads, so scanning only the busiest heads leaves about half of them.
- Recent memory covers 22-23% of the missed attention. Position explains almost none.
- A hindsight pick of 13 blocks would cover 56-63%. The information is in the query, so only a scan finds it.

Rules, amendments and full tables are in `docs/JUMPS.md`.

## Checks

- **Rules before data.** The decision rule was frozen before the deciding runs. The jump-test rule and code were committed (82c1030) before any full-run data.
- **Second PyTorch version.** The Granite runs were repeated under torch 2.15.0. Same verdict; pooled numbers moved by at most 0.13 points.
- **Controls on the jump test.** The negative control passes. The positive control failed as defined and is reported as failed; token alignment is covered by a unit test instead.
- **Backend fault.** Under torch 2.9.0 the MPS backend intermittently returned wrong results in plain decoding. `tools/mps_decode_repro.py` reproduces it. It did not recur under 2.15.0.

## Limits

- 4K contexts only. The scan cost matters most at longer contexts, which are untested.
- Three small models from two families.
- Teacher-forced perplexity only; no task scores from free-running generation.
- Document-continuation text; copy-heavy output such as code was not tested.
- All runs on Apple's MPS backend. Nothing was run on CUDA.
- Per-prompt figures are noisy at 128 scored tokens; only pooled figures are reliable.
- The comparison contains only simple scalar codecs. ReadKV, sparse top-k methods and stronger codecs were not run here.

## Repo layout

| Path | Purpose |
| --- | --- |
| `phase0/capture.py` | Records block attention mass per head and step |
| `phase0/analyze.py` | Concentration and stability analysis |
| `phase0/tiered.py` | Simulator for the tiered variants and key codecs |
| `phase0/analyze_d.py` | Bootstrap intervals and the decision rule |
| `phase0/compare_runs.py` | Compares runs across environments against pass criteria |
| `phase0/benchmarks.py` | LongBench and RULER loaders and scorers |
| `phase0/jumps_capture.py`, `phase0/jumps_analyze.py` | Jump test |
| `scripts/phase0_run.sh` | Single-GPU runner |
| `tools/mps_decode_repro.py` | Reproduces the MPS fault |
| `tests/` | Unit tests |
| `docs/JUMPS.md` | Jump-test rules, amendments and results |
| `results/prescreen-4k/` | Result summaries per model |
| `results/jumps-4k/` | Jump-test verdict and tables |
| `data/prompts/` | Prompt sets |
| `PHASE0.md` | Experiment spec |

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # pins torch 2.15.0
pytest
```

The runner works in stages and needs a model cache directory:

```bash
export HF_HOME=/path/to/model/cache
STAGE=replicate4k bash scripts/phase0_run.sh
```

Stages: `replicate4k`, `verify`, `timing`, `uniform`, `ladder`, `all`. Runs are resumable, and a `MAX_HOURS` gate (default 1) guards against long stages.

On Apple silicon, use torch 2.15.0 or later. See `PHASE0.md` for the experiment spec and `docs/JUMPS.md` for the jump test.

A run on an NVIDIA GPU with Qwen2.5-7B at 8K to 32K is prepared (prompt sets and runner stages) but has not been done.

## Related work

- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180): the paged allocator this work starts from.
- [ReadKV](https://arxiv.org/abs/2610.11245): read-time precision chosen per query; the method closest to variant f.
- [SPIN](https://arxiv.org/abs/2610.09025): predicts important blocks from score history in models with a trained indexer.
- [RazorAttention](https://arxiv.org/abs/2407.15891): ranks heads by how they follow repeated tokens.
- [Lost in a Few Heads](https://arxiv.org/abs/2611.21917): block-selection failures concentrated in a few heads of sparse-attention models.

## How this was built

The experiments were designed and analysed in conversation with Claude, and the code was written and run with Claude Code. Decision rules were written down before the data they judge.

## Full report

A longer account, including the original proposal, every result table and the mistakes corrected along the way, is in `docs/REPORT.md`.
