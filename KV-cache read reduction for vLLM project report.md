# KV-cache read reduction for vLLM: project report

Nov 18, 2026 · @Thaneesh

## Summary

The project confirmed a real saving in KV-cache bytes read, but the method that delivers it was published first as ReadKV, and both of our own new ideas failed.

- **Confirmed.** Reading the cache in three precision tiers on every decode step uses 30-34% fewer bytes than the best simple compressed cache, with perplexity within 1%. This held on three small models from two families at 4K tokens.
- **Not new.** The surviving design is ReadKV's per-step loop in a simpler block-level form. Eight further ideas screened on 18 November were already published.
- **Failed.** Stored read plans added nothing. Predicting what to read from repeated tokens was no better than random picks. Scanning only a few attention heads does not work either.
- **Not measured.** No speed was measured. Bytes were counted in simulation on a laptop. Nothing was run above 4K tokens, on a model larger than 3B, or on an NVIDIA GPU.
- **Status.** The search for a new mechanism stopped on 18 November 2026. Code, results and decision rules are committed in the repo. A write-up has not been started.

## Goal and background

The goal was a new way to raise vLLM throughput beyond PagedAttention, as a portfolio project.

- **Original request, 15 November 2026.** Study how GPUs work internally, research the field, and propose a new algorithm or an improvement on PagedAttention.
- **Why bytes read is the yardstick.** Decoding on a GPU is limited by memory bandwidth, not compute. An H100 offers about 295 operations per byte of bandwidth. Decode attention uses about 0.5 to 1 per byte, or 4 with grouped-query attention.
- **What PagedAttention left open.** It fixed how cache memory is allocated. Every block is still read in full, at its stored precision, on every step.

All results below are stated as a fraction of the bytes a full-precision (fp16) cache reads per step, at equal output quality.

## Timeline

The whole project ran over four days, 15 to 18 November 2026. Newest first.

| Date | Step | Outcome |
| --- | --- | --- |
| 18 Nov | Rented-GPU question | Advice given: skip the run for now. It would not change the story, and the runner stays ready |
| 18 Nov | Jump test, full run | NO-GO on all three models. Results committed as f51e80c |
| 18 Nov | Jump test, controls | Positive control failed as defined. Token alignment verified separately |
| 18 Nov | Paper check: Lost in a Few Heads | Not prior art for the idea. It showed a cheap competitor, so the test gained a concentration check |
| 18 Nov | Second prior-art pass on lexical triggers | No direct prior art found. The premise is published (RazorAttention). Test redesigned |
| 18 Nov | Idea screen | Nine candidates. Eight already published. Lexical triggers survived the search |
| 18 Nov | Prior-art check on the surviving design, then a second pass | The core loop is ReadKV's. Block-level precision tiers also exist elsewhere |
| By 17 Nov | Granite replication and backend fault | Third model passes. An MPS fault was found and bounded. A rerun on torch 2.15.0 agrees. Commit 5ae34a4 |
| 15-17 Nov | Phase 0 runs on Qwen 1.5B and 3B | A 4-bit cache is the real bar. Decision rule frozen. Variants e and f pass on the 4/3 ladder |
| 15-17 Nov | Infrastructure | Northeastern's Explorer cluster ruled out for a personal project. A Vast.ai rental was scoped and never used |
| 15 Nov | Handoff to Claude Code | CLAUDE.md and PHASE0.md written. The laptop was chosen for Phase 0 |
| 15 Nov | Proposal | Read-plan paging written up after a literature review |

Exact days between 15 and 17 November are not recorded. The results log was last updated on 17 November.

## The original proposal

The proposal, read-plan paging, kept vLLM's blocks and gave each sequence a stored plan for how much of each block to read. Only its measurement phase was carried out.

- **Design.** A second, low-precision copy of every block. Beside each sequence's block table sits a 2-bit read plan per block, layer and KV head: HOT (read in full), WARM (read the low-precision copy) or COLD (skip). Plans are reused across steps and refreshed every R steps.
- **Tiering rule.** Promote blocks to HOT by squared attention weight until what remains is under a budget. Mark as COLD the lowest-mass blocks that together hold less than a small share of attention.
- **Expected gain.** A bandwidth-only estimate gave 1.4x to 1.8x over an fp8 cache at batch 24 with 16K contexts on an H100. This was an estimate, not a measurement.
- **Go criterion as first written.** Read at most 0.30 of fp16 bytes at no more than 1% perplexity increase, with a refresh interval of at least 4 steps.
- **Build plan.** Phase 0 measure, Phase 1 reference implementation, Phase 2 vLLM plugin with timing, Phase 3 stretch items. Phases 1 to 3 were not started.

The proposal listed four things as new. None held up.

| Claimed as new | What happened |
| --- | --- |
| Read plan stored in the per-sequence block table | Dropped with stored plans. Recording tiers per page also exists in Minima-KV and a GitHub project |
| Three tiers under one error-budget rule | Works, but ReadKV already sets value precision from squared attention, at finer levels |
| Plan reuse across steps | Refuted by our data. Also published: SPIN, AttentionPredictor, Recycled Attention |
| A test in batched, paged serving | Never reached |

## How the experiments were run

Everything was simulated on a MacBook: the model runs normally, and the experiment counts the cache bytes each variant would read.

| Item | Value |
| --- | --- |
| Models | Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, IBM Granite 3.3 2B Instruct (40 layers, 32 query heads, 8 KV heads, head dim 64) |
| Hardware | MacBook M2 Pro, 32 GB, PyTorch MPS backend, fp16 |
| Software | torch 2.9.0 for the original runs; torch 2.15.0 in a separate environment for later runs |
| Data | 40 LongBench documents cut at 4,096 tokens, six document types, re-tokenised per model |
| Scoring | The next 129 tokens, teacher-forced: 5,120 scored tokens per model |
| Quality metric | Perplexity change against fp16, with a bootstrap 95% interval over prompts |
| Bytes read | Fraction of fp16 KV bytes read per step, quantiser metadata included |
| Block size | 16 tokens, as in vLLM |
| Tiering settings | eps\_w = 1e-3, eps\_c = 0.02, R = 4 where a plan is stored |

**Variants tested.**

| Variant | What it does |
| --- | --- |
| a | Fresh plan from exact attention on every step (an oracle) |
| b | Stale plan, no check |
| c | Stale plan plus same-step promotion from a 1-bit key scan |
| d | Two tiers, no COLD |
| e | Stale plan; keys of COLD blocks are still read every step |
| f | No stored plan: read all keys at the WARM width each step, apply the rule, read values by tier |

**Key-scale codecs.** Scales per 16-token block (heavy metadata), scales shared over 256 tokens (group256), or scales calibrated once per layer and head (static).

**Precision ladders (HOT/WARM bits).** The first ladder was fp16/4-bit. It was re-based to 4/3 and 4/2 once 4 bits proved to be the lowest uniform width within 1%.

**Decision rule, frozen before the deciding runs.** GO if variant e or f stays within 1% perplexity and reads at least 25% fewer bytes than the cheapest uniform cache within 1%. Caches with different key and value widths count toward that bar.

The rule was revised several times before it was frozen, each time before the ladder data existed. The Qwen 3B ladder run, the Qwen 1.5B run of variant f and the whole Granite run came after the freeze.

**Who did what.** The design, analysis and prior-art checks were done in chat with Claude. The code was written and run by Claude Code on the laptop, from two handoff files. The repo was not inspected from the chat; its contents are known from Claude Code's reports.

## Results: simple compressed caches

A plain 4-bit cache is the real competitor: it reads about a quarter of the fp16 bytes and stays within 1% of fp16 quality.

**Qwen2.5** (key scales shared over 256 tokens unless noted; perplexity change against fp16 with 95% interval):

| Cache | Bytes read | Qwen 1.5B | Qwen 3B |
| --- | --- | --- | --- |
| Uniform 8-bit | 0.51-0.57 | about 0% | about 0% |
| Uniform 4-bit, per-block scales | 0.323 | passes | +0.30% \[0.10, 0.51\] |
| **Uniform 4-bit** | **0.264** | **+0.55% \[0.25, 0.84\]** | **+0.61% \[0.24, 0.99\]** |
| Uniform 4-bit, static scales | 0.260 | +1.85% (fails) | +0.98% \[0.45, 1.53\] |
| Keys 4-bit, values 3-bit | 0.233 | +1.59% \[1.11, 2.08\] | +1.40% \[0.96, 1.90\] |
| Keys 3-bit, values 4-bit | 0.233 | +2.37% \[1.64, 3.19\] | +1.42% \[0.72, 2.01\] |
| Uniform 3-bit | 0.202 | +3.38% \[2.72, 4.11\] | +1.82% \[0.99, 2.63\] |
| Keys 4-bit, values 2-bit | 0.202 | +6.92% \[5.63, 8.20\] | +4.26% \[3.16, 5.44\] |
| fp8 e4m3 cast (emulation only) | 0.500 | +277% | +10.5% \[6.7, 14.9\] |

**Granite 3.3 2B** (fp16 perplexity 4.876, torch 2.9.0 run):

| Cache | Key scales | Bytes read | Perplexity change | Within 1% |
| --- | --- | --- | --- | --- |
| Uniform 8-bit | group256 | 0.521 | +0.02% \[-0.01, 0.05\] | yes |
| Uniform 4-bit | per-block | 0.331 | +0.19% \[-0.05, 0.46\] | yes |
| Uniform 4-bit | group256 | 0.272 | +0.20% \[-0.14, 0.55\] | yes |
| Uniform 4-bit | static | 0.268 | +0.68% \[0.03, 1.32\] | yes (interval top above 1%) |
| **Keys 4-bit, values 3-bit** | **group256** | **0.241** | **+0.30% \[-0.08, 0.71\]** | **yes (the bar)** |
| Keys 3-bit, values 4-bit | group256 | 0.241 | +1.18% \[0.40, 2.04\] | no |
| Keys 4-bit, values 2-bit | group256 | 0.210 | +1.10% \[0.42, 1.88\] | no |
| Uniform 3-bit | group256 | 0.210 | +1.36% \[0.55, 2.18\] | no |

**The bar per model** (cheapest uniform cache within 1%): 0.264 on Qwen 1.5B, 0.260 on Qwen 3B (static-scale 4-bit), 0.241 on Granite (keys 4-bit, values 3-bit).

What these runs show:

- 4 bits is the lowest symmetric width within 1% on all three models. 3-bit fails everywhere.
- The bar depends on the model. Keys at 4 bits with values at 3 bits fails on both Qwen models and passes on Granite.
- Scale metadata matters. Per-block scales cost 0.32-0.33, against 0.26-0.27 for scales shared over 256 tokens.
- The fp8 result is an emulation and was not reproduced in real vLLM. It tracks key magnitude: the largest key is about 300 on Qwen 1.5B, 96 on Qwen 3B and 116 on Granite.

## Results: tiered reads

Variants e and f pass the frozen rule on all three models with the 4/3 ladder, reading 30-34% fewer bytes than the bar.

**Deciding runs** (original runs, torch 2.9.0):

| Model | Variant | HOT/WARM bits | Perplexity change | Bytes read | Below bar | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen 1.5B | e | 4/3 | +0.47% \[0.15, 0.80\] | 0.174 | 34% | pass |
| Qwen 1.5B | f | 4/3 | +0.47% \[0.16, 0.74\] | 0.176 | 33% | pass |
| Qwen 3B | e | 4/3 | +0.65% \[0.25, 1.06\] | 0.179 | 31% | pass (interval top above 1%) |
| Qwen 3B | f | 4/3 | +0.63% \[0.22, 1.03\] | 0.181 | 30% | pass (interval top above 1%) |
| Granite 2B | e | 4/3 | +0.19% \[-0.22, 0.63\] | 0.159 | 34% | pass |
| Granite 2B | f | 4/3 | +0.40% \[0.00, 0.81\] | 0.161 | 33% | pass |
| Qwen 1.5B | e | 4/2 | +0.93% \[0.46, 1.40\] | 0.129 | 51% | pass (interval top above 1%) |
| Qwen 1.5B | f | 4/2 | +1.64% \[0.99, 2.30\] | 0.127 | 52% | fails quality |
| Qwen 3B | e | 4/2 | +1.25% \[0.71, 1.79\] | 0.133 | 49% | fails quality |
| Qwen 3B | f | 4/2 | +1.82% \[1.08, 2.60\] | 0.131 | 50% | fails quality |

The 4/2 ladder was not run on Granite.

**Other variants** (Qwen 1.5B only, not deciding):

| Variant | HOT/WARM bits | Perplexity change | Bytes read | Verdict |
| --- | --- | --- | --- | --- |
| c | 4/3 | +0.80% \[0.43, 1.21\] | 0.175 | pass |
| d | 4/2 | +1.10% \[0.68, 1.54\] | 0.152 | fails quality |
| c | 4/2 | +7.37% \[5.90, 8.96\] | 0.132 | fails quality |

**Variant f against e on the 4/3 ladder:**

| Measure | Qwen 1.5B | Qwen 3B | Granite 2B |
| --- | --- | --- | --- |
| f's HOT set: recall against the oracle plan | 87% | 88% | 90% |
| f's HOT set: precision against the oracle plan | 83% | 83% | 85% |
| Oracle-HOT blocks that f marks COLD | 0.003% | 0.006% | 0.037% |
| Query heads with over 5% of attention on COLD blocks: f | 0.004% | 0.01% | 0.07% |
| Same, for e after fix-ups | 3.2% | 2.1% | 6.7% |
| Steps where e needs a fix-up | 2.3% | 2.4% | 1.1% |

On Granite the planned split for f is 3% HOT, 50% WARM and 47% COLD.

What these runs show:

- **About a third fewer bytes at equal quality.** Tiered reads use 0.16-0.18 of fp16 bytes, against 0.24-0.26 for the best simple cache.
- **Stored plans add nothing.** Variant f keeps no plan and matches e on quality and bytes.
- **The 4/2 ladder is too coarse.** It halves the bytes but breaks the 1% limit, except for one marginal pass.
- **The Qwen 3B passes are marginal.** Their interval tops are 1.06% and 1.03%.
- **Estimated speed effect.** On a bandwidth-only estimate at batch 24 with 16K contexts, a third fewer bytes is roughly 1.15x to 1.2x throughput over a 4-bit cache. This was never measured.

## Checks and replications

The laptop results survived three checks, but nothing has been run on an NVIDIA GPU.

**Granite rerun on torch 2.15.0** (same 40 prompts, MPS):

| Variant | torch 2.15.0 | torch 2.9.0 | Bytes read |
| --- | --- | --- | --- |
| Keys 4-bit, values 3-bit (the bar) | +0.29% \[-0.09, 0.70\] | +0.30% \[-0.08, 0.71\] | 0.241 |
| Uniform 4-bit | +0.14% \[-0.19, 0.50\] | +0.20% \[-0.14, 0.55\] | 0.272 |
| e, 4/3 | +0.11% \[-0.29, 0.54\] | +0.19% \[-0.22, 0.63\] | 0.159 |
| f, 4/3 | +0.27% \[-0.11, 0.68\] | +0.40% \[0.00, 0.81\] | 0.161 |

Same bar, same bytes read, same tier fractions, same verdict. Pooled changes moved by at most 0.13 points.

**Backend fault on MPS.**

- Under torch 2.9.0, stock PyTorch decoding of Granite on MPS is intermittently wrong: NaN in 2 of 6 plain fp16 runs, and one fp32 run that returned perplexity 4.08 instead of 1.458 with no NaN.
- It is a timing or buffer-reuse fault in the backend, not fp16 overflow as first recorded. Garbage values appear in the new token's query and key before attention runs.
- Under torch 2.15.0 it did not reproduce in 15 runs.
- Checks made on the committed results: a scan of every per-prompt value on all three models found no non-finite values and nothing beyond 10% in a deciding row. A one-prompt rerun under 2.9.0 reproduced the committed numbers exactly.
- The two Qwen models were not rerun on torch 2.15.0.

**Per-prompt numbers are noisy.**

- Between the two torch versions, 81 of 200 per-prompt rows differ by more than 0.3 points. The median difference is 0.25-0.35 points and the largest is 1.85.
- With 128 scored tokens per prompt, a few changed tokens move a prompt by about a point. The likely cause, not verified, is tiny fp16 differences crossing quantiser rounding boundaries.
- Only pooled figures are reliable. Worst-prompt and prompts-above-2% figures are indicative only; the full table is in the project's results log.

## Prior art

The design that survived Phase 0 is the per-step loop of [ReadKV](https://arxiv.org/abs/2610.11245), published on 8 October 2026 and tested on more and larger models.

**ReadKV against variant f:**

|  | ReadKV | Variant f |
| --- | --- | --- |
| Per-step loop | Read keys, rebuild attention, then choose how much of each value to read | Same |
| Importance signal for values | Sum of squared attention over the query heads sharing a KV head | Same |
| Key reading | Depth chosen per channel from the query, 0 to 8 bits | Every key at 3 bits; top-tier keys re-read at 4 |
| Value levels | Any depth from 0 to 8 bits per token | Three levels: 4 bits, 3 bits, unread |
| Granularity | Per token | 16-token blocks |
| Reuse across steps | Not tested | Tested and dropped |
| Models | Qwen2.5 3B/7B/14B, Yi-1.5 6B, DeepSeek-LLM 7B, Mistral 7B | Qwen2.5 1.5B/3B, Granite 3.3 2B |
| Context lengths for quality | 2K to 8K | 4K |
| Headline quality result | Within 0.66% of dense perplexity reading 4 of 16 bits | Within 1% reading 0.16-0.18 of fp16 bytes, metadata counted |
| Its own uniform 4-bit baseline | Loses 1.3% to 3.4% | Loses 0.2% to 0.6% |
| Serving throughput | Not tested | Not tested |

The two sets of numbers cannot be compared. Models, data, context lengths and codecs all differ. Whether variant f is better or worse than ReadKV's reader is unknown.

**Block-level precision tiers also exist:**

| Work | What it does | How it differs from variant f |
| --- | --- | --- |
| [Xie et al.](https://arxiv.org/abs/2503.18869) (Mar 2025) and [TRACE](https://arxiv.org/abs/2509.03377) | 16-token pages: top pages at BF16, the next at FP8 or FP4, the rest dropped | A sketch in a hardware paper, with one perplexity table on one model |
| [Minima-KV](https://arxiv.org/pdf/2608.23834) (Aug 2026) | Paged runtime with pages in FP8 or 3-bit, per-format kernels merged by a global softmax | One format per page, set by rewriting the page; attention scoring disabled in its evaluations |
| [Semantic-Aware Adaptive KV Cache Precision](https://github.com/lecuong1502/Semantic-Aware_Adaptive_KV_Cache_Precision/issues/88) (GitHub, Oct 2026) | Per-page attention mass drives downgrades of the least-attended pages | Rewrites stored pages; the goal is memory, not bytes read |

**Other close work:**

| Work | What it does |
| --- | --- |
| [Fathom](https://arxiv.org/pdf/2609.17652) (Sep 2026) | Scans a 4-bit copy of all keys, fetches the top tokens in full, leaves the rest unread; built for offloaded caches, tested to 1M tokens |
| [NOVA-KV](https://arxiv.org/pdf/2608.04074) (Aug 2026) | Stores the cache at about 2.2 bits per element; 1.6x to 3.1x throughput over BF16 in an SGLang fork on one H100, with task scores down 1 to 4 points on average |
| [Cocktail](https://arxiv.org/abs/2503.23294) (Mar 2025) | Chunk-level bit widths chosen once from query-to-chunk similarity |
| [HyQuant](https://arxiv.org/pdf/2608.27875) (Aug 2026) | Low-bit cache with full precision kept for selected keys and a recent window |
| [COBS](https://arxiv.org/pdf/2607.09052) (Jul 2026) | Block-sparse attention from per-block summaries, with full reads for the top blocks |

**What this project adds, as far as the search found.** Evidence, not a mechanism: a per-step, three-level read held to a 1% quality bar on three models, with metadata counted and a fixed rule.

**Gaps in our comparison.** The bar contains only simple scalar codecs. Stronger codecs such as NOVA-KV, sparse top-k methods and ReadKV itself were never run in our harness.

## Idea screen and the lexical-trigger test

Nine fresh ideas were screened on 18 November. Eight were already published, and the ninth failed its test the same day.

**The screen:**

| # | Idea | Already published as |
| --- | --- | --- |
| 1 | Predict the important blocks from earlier steps (our original plan) | [SPIN](https://arxiv.org/abs/2610.09025) (Oct 2026), [AttentionPredictor](https://arxiv.org/pdf/2502.04077) (2025), [Recycled Attention](https://arxiv.org/html/2411.05787v1) (2024) |
| 2 | Reuse one layer's block choice in later layers | [IndexCache](https://github.com/THUDM/IndexCache), [HySparse](https://arxiv.org/html/2602.03560), [LatentIndex](https://arxiv.org/html/2610.04635) |
| 3 | Read more only when the model looks unsure | [Conf-KV](https://arxiv.org/html/2605.24786v1); Don't Waste Bits (known second-hand) |
| 4 | Skip heads that only look at the first token | [SinkRouter](https://arxiv.org/pdf/2604.16883) |
| 5 | Cheap draft read, full check every few tokens | [QuantSpec](https://arxiv.org/pdf/2502.10424) (ICML 2025) |
| 6 | Guess the next step's blocks from a draft token | [OasisKV](https://arxiv.org/pdf/2608.08097), [DualDecoder](https://arxiv.org/pdf/2607.26475) |
| 7 | Reuse a selection when consecutive queries are similar | [TokenSelect](https://aclanthology.org/2025.emnlp-main.1079.pdf) |
| 8 | Read a shared prefix once for many requests | [Prefix-aware attention](https://arxiv.org/pdf/2505.17694) |
| 9 | Lexical triggers: choose blocks to read from token matches | Not found in about twenty searches. Tested and rejected, below |

For scale, SPIN's history-based predictor gained 10.6% to 14.9% throughput in vLLM at a 507K-token context. An [NVIDIA design guide](https://developer.nvidia.com/blog/co-designing-sparse-attention-for-efficient-long-context-inference) from 17 November covers only models with sparse attention trained in, and sets aside drop-in methods for ordinary models.

**The lexical-trigger idea.** The server knows the next input token before the GPU step. A lookup from token to blocks, built when the prompt is first processed, names blocks where that token appeared before. Read those blocks in place of random picks, and skip the full key scan.

**Nearest neighbours found, none of them this idea:**

| Work | What it does |
| --- | --- |
| [RazorAttention](https://arxiv.org/pdf/2407.15891) (ICLR 2025) | Ranks heads once, offline, by how strongly they follow repeated tokens; keeps a full cache for the top heads |
| [Lost in a Few Heads](https://arxiv.org/pdf/2611.21917) (16 Nov 2026) | In models with block-sparse attention trained in, a few heads pick the wrong blocks; fixes them with full attention or a 4-bit key scan |
| [Latest Exact Match Attention](https://arxiv.org/pdf/2609.25802) (Sep 2026) | A new architecture, trained from scratch, that reads by hash lookup on learned binary keys |
| [Transactional Attention](https://arxiv.org/pdf/2604.11288) (Apr 2026) | Text patterns such as a key label protect the next tokens from eviction |

**The test.** 40 prompts per model at 4K, using the stale plan of variant e. A jump is a head-step with more than 5% of attention on blocks the plan marked COLD. Heads were ranked on half the prompts and everything was scored on the other half. Recent memory picked first (5% of blocks), then lexical or random picks (another 5%, 13 blocks).

**Rules, committed before the data.** NO-GO if the top 10% of heads hold at least 90% of the missed attention. Otherwise GO if, outside those heads, lexical picks cover at least 40% of what is left and at least 5 times what random picks cover. Otherwise NO-GO.

**Result: NO-GO on all three models** (held-out half, heads outside the top 10%):

| Model | Top 10% of heads hold | Lexical | Random | Best possible 13 blocks | Jumps with one block holding half |
| --- | --- | --- | --- | --- | --- |
| Qwen 1.5B | 32.3% | 14.1% | 14.8% | 60.7% | 4.1% |
| Qwen 3B | 33.1% | 14.0% | 15.3% | 62.7% | 5.1% |
| Granite 2B | 35.2% | 10.9% | 9.7% | 56.2% | 3.3% |

- Recent memory covers 22-23% of the missed attention on every model. Fully scanning the top 10% of heads still leaves 48-52% missed.
- Position explains little: about 1% sits in the first 4 blocks, and 2% to 8% in the 8 blocks behind the recent window.
- Over all prompts and heads, 5-7% of jumps have one block holding half the missed attention.
- Matching on token text instead of token IDs gives the same verdict.

**Checks on the test.**

- The negative control passes: a lookup built from another prompt scores 1.05 to 1.24 times count-matched random.
- Jump counts match the committed variant-e runs to within 0.13%.
- The positive control failed as defined. On repeated random tokens, lexical covered 42.4%, 12.6% and 8.4% against 13.2%, 16.0% and 5.4% for random, with 90% required. Token alignment was verified separately. The control was badly designed: the stale plan already keeps the copy target most of the time.

**What the result says.** What a stale plan misses is spread over many heads and several blocks. It can be found, but only by scanning the keys with the current query. That supports the per-step scan of variant f for ordinary models at a tight quality bar, at least at 4K.

## What failed

Seven things did not work or did not hold up.

| What | How it failed | Evidence |
| --- | --- | --- |
| Stored read plans (the original idea) | Added nothing. Deciding fresh on every step did as well | Variant f matches e on quality and bytes on all three models |
| Lexical triggers (the second idea) | No better than random picks outside the busiest heads | 11-14% against 10-15%; 40% was needed |
| Scanning only a few heads | The misses are spread over many heads | Top 10% of heads hold about a third; half is still missed after scanning them |
| The 4/2 ladder | Saves about half the bytes but breaks the 1% limit | Fails on 3 of 4 runs; the one pass has its interval top above 1% |
| Novelty | The working design is ReadKV's; eight other ideas were published | Prior-art checks of 18 November |
| The first gain estimate | 1.4x to 1.8x over fp8 shrank to about 1.15x to 1.2x over a 4-bit cache | The real competitor is 4-bit, and tiered reads save a third, not more |
| The positive control of the jump test | Could not reach its 90% threshold as defined | The stale plan already keeps the copy target most of the time |

## Mistakes and corrections

Thirteen mistakes were made and corrected along the way. Most were Claude's, in chat.

| Mistake | Made by | Correction |
| --- | --- | --- |
| First tiering rule (promote until X% of attention is covered) marked three quarters of a diffuse head HOT | Claude, chat | Replaced by the squared-weight budget rule. Stochastic rounding was also dropped as worse than nearest |
| PHASE0 spec stored the maximum over query heads while the rule tiers per query head | Claude, chat | Flagged by Claude Code. Per-query-head storage adopted |
| Squared block mass underflowed in float16 at long contexts | Caught by Claude, chat | Store its square root instead |
| Go rule used an absolute target (read at most 0.30) that ignored the 4-bit competitor | Claude, chat | Made relative to the cheapest passing uniform cache. Revised several times, then frozen before the deciding data |
| Expected gain of 1.4x to 1.8x over fp8 | Claude, chat | Revised to about 1.15x to 1.2x over a 4-bit cache, still only an estimate |
| Plan reuse claimed as new and useful | Claude, chat | Refuted by variant f; later found published |
| MPS fault blamed on fp16 overflow in the attention kernel | Claude Code, repeated by Claude | Bisection showed an intermittent timing or buffer-reuse fault. Log corrected |
| A 0.3-point per-prompt flag, and a tail comparison between variants | Claude, chat | Both were noise. Tail figures are labelled indicative only; the 7B prompt sets were rebuilt with 257 reference tokens |
| First prior-art search (15 Nov) missed Minima-KV, details of Xie et al., Fathom, a GitHub project, SPIN, AttentionPredictor and Recycled Attention | Claude, chat | Found in later passes on 18 November |
| First prior-art answer credited the project with block-level decisions and called the result comparable to ReadKV's and a replication | Claude, chat | Withdrawn on reflection the same day |
| First version of the jump test used fixed block budgets, which diffuse heads would dominate | Claude, chat | Redesigned to count jumps directly, with a concentration check |
| Positive control measured the wrong thing | Claude, chat | Recorded as failed. Not changed a third time. Alignment is covered by a unit test |
| Misses described as attention jumping to one block | Claude, chat | Only 3-5% look like that outside the busiest heads, 5-7% overall. Noted in docs/JUMPS.md |

Two amendments by Claude Code were made before the full jump run and logged: the negative control compares against count-matched random, and the heads that fire are those whose misses concentrate in one block.

## Limits, and what was not done

Every result is a byte count from simulation on small models at 4K tokens, so the saving is unproven at realistic scale and as speed.

**Limits of what was measured.**

- Three small models (1.5B to 3B) from two families, 4K contexts only.
- Teacher-forced perplexity only. No task scores from free-running generation.
- Document-continuation text. Copy-heavy output such as code was not tested.
- All runs on the MPS backend. Granite was reproduced on a second torch version; the two Qwen models were not.
- The Qwen 3B passes are marginal, with interval tops of 1.06% and 1.03%.
- Variant e refreshes its plans from exact attention weights, a shortcut f does not have.
- The jump test covers a tail: 2.1% to 6.7% of head-steps under variant e.

**Not done.**

| Item | State |
| --- | --- |
| Rerun of one 4K set on CUDA | Runner stage built (replicate4k), never run |
| Qwen2.5-7B at 8K to 32K | Prompt sets and runner built, never run |
| Free-run task scores (LongBench, RULER) | Loaders and scorers built, never run |
| Sparse top-k, ReadKV and strong-codec baselines | Not built |
| Real serving throughput in vLLM | Not started (Phase 2 of the proposal) |
| A real kernel and storage layout for read-time precision | Not started |
| Check of the fp8 result in real vLLM | Not done |
| The parked idea: block statistics pooled across requests sharing a prefix | Untested, and the jump result lowers its odds |

**Still open in the literature, as far as the searches found.** A per-step multi-level read judged at a tight quality bar above 8K tokens, and end-to-end serving speed for read-time precision.

## Decisions and current status

The project is paused at a finished 4K study, with no write-up and no cloud spend recorded.

| Decision | When | Basis |
| --- | --- | --- |
| Run Phase 0 on the MacBook M2 Pro, not a rented GPU | 15 Nov | Small models at 4K fit in 32 GB |
| Build and run with Claude Code from two handoff files | 15 Nov | Keeps the code and runs on the laptop |
| Do not use Northeastern's Explorer cluster | 15-17 Nov | Not allowed for a personal project |
| Rental plan: one 48 GB RTX A6000 on Vast.ai, about $0.28 to $0.50 an hour | 15-17 Nov | Scoped, never used |
| Freeze the go rule before the deciding runs | 15-17 Nov | Avoids tuning the rule to the data |
| Drop stored plans | By 17 Nov | Variant f matches e |
| Move new runs to torch 2.15.0 | By 17 Nov | The MPS fault does not reproduce there |
| Stop the search for a new mechanism | 18 Nov | Two own ideas refuted, eight others published |
| Rented-GPU run at 8K to 32K | 18 Nov | Not done. Advice was to skip it for now; no final choice is recorded |

**Open choices.**

- Whether to write up the 4K study as it stands. An outline was offered and not yet requested.
- Whether to do the rented-GPU run later. It would extend the evidence to a 7B model at longer contexts, and would not change the main conclusions.

**Known stale spots in the project notes.** The results log still lists the GPU runs as still to come and still describes the misses as jumps to one block.

## Where everything lives

The code and results are in the project repo on the laptop, and the working notes are in the claude.ai project. Repo details are as reported by Claude Code.

**Commits:**

| Commit | What it holds |
| --- | --- |
| f51e80c | Jump-test results and write-up. Working tree clean |
| e763cfc | Three extra reported numbers for the jump test, logged before the full run |
| 82c1030 | Jump-test code, rule and both amendments, before any full-run data |
| 5ae34a4 | End of Phase 0: single-GPU runner, pass rule for the CUDA check, 257-token 7B prompt sets; 107 tests |

At the last report, all 114 tests pass.

**Repo paths:**

| Path | Purpose |
| --- | --- |
| `phase0/capture.py` | Records block attention mass per head and step |
| `phase0/analyze.py` | Concentration and stability analysis |
| `phase0/tiered.py` | Simulator for variants a to f and the key codecs |
| `phase0/analyze_d.py` | Bootstrap intervals and the go rule |
| `phase0/compare_runs.py` | Compares runs across environments against pass criteria |
| `phase0/benchmarks.py` | LongBench and RULER loaders and scorers |
| `phase0/jumps_capture.py`, `phase0/jumps_analyze.py` | Jump-test capture and analysis |
| `scripts/phase0_run.sh` | Single-GPU runner: replicate4k, verify, timing, uniform, ladder |
| `tools/mps_decode_repro.py` | Reproduces the MPS fault |
| `tests/` | Unit tests, including `tests/test_jumps.py` |
| `docs/JUMPS.md` | Jump-test rules, amendments, results |
| `results/prescreen-4k/` | Phase 0 summaries: qwen2.5-1.5b, qwen2.5-3b, granite-3.3-2b, granite-3.3-2b-torch2.15 |
| `results/jumps-4k/decision.json` | Jump-test verdict and tables |
| `data/prompts/` | 4K prompt sets, and the 7B sets: 30 prompts each at 8,192, 16,384 and 32,000 tokens |
| `runs/` | Raw outputs. Git-ignored, so they exist only on the laptop |
| `CLAUDE.md`, `PHASE0.md` | Working rules and the Phase 0 spec |
| `requirements.txt` | Pins the torch 2.15.0 environment (`~/venvs/rpp-torchlatest`); the base conda environment stays on 2.9.0 |

**Project notes** (claude.ai project AI project ideas):

| Note | Contents |
| --- | --- |
| `claude/read-plan-paging-kv-cache.md` | The original proposal, 15 November. Superseded on novelty and on the competitor |
| `claude/phase0-results-log.md` | Full Phase 0 results, last updated 17 November |
| `claude/prior-art-check-2026-11-18.md` | ReadKV comparison and related work |
| `claude/idea-screen-2026-11-18.md` | The idea screen and the jump-test verdict |

## Sources

Experimental numbers come from Claude Code's run reports, as transcribed into the project notes. Papers were read to different depths, listed here.

**Opened and read, in full or through a tool-assisted read:**

- [ReadKV: Read What Matters](https://arxiv.org/abs/2610.11245)
- [Xie et al., Reimagining Memory Access for LLM Inference](https://arxiv.org/abs/2503.18869)
- [Minima-KV](https://arxiv.org/pdf/2608.23834)
- [NOVA-KV](https://arxiv.org/pdf/2608.04074) (the last quarter was not read)
- [Semantic-Aware Adaptive KV Cache Precision](https://github.com/lecuong1502/Semantic-Aware_Adaptive_KV_Cache_Precision/issues/88) (one issue page)
- [SPIN](https://arxiv.org/abs/2610.09025)
- [NVIDIA, Co-Designing Sparse Attention](https://developer.nvidia.com/blog/co-designing-sparse-attention-for-efficient-long-context-inference)
- [Lost in a Few Heads](https://arxiv.org/pdf/2611.21917) (PDF supplied by the user; main text read)
- [Latest Exact Match Attention](https://arxiv.org/pdf/2609.25802)
- [Transactional Attention](https://arxiv.org/pdf/2604.11288)
- [sparse-kv-reads](https://github.com/aaholmes/sparse-kv-reads)
- [KV-cache survey](https://arxiv.org/abs/2607.02574)

**Known from abstracts, search excerpts or earlier notes:**

- [TRACE](https://arxiv.org/abs/2509.03377), [Fathom](https://arxiv.org/pdf/2609.17652), [Cocktail](https://arxiv.org/abs/2503.23294), [HyQuant](https://arxiv.org/pdf/2608.27875), [COBS](https://arxiv.org/pdf/2607.09052)
- [AttentionPredictor](https://arxiv.org/pdf/2502.04077), [Recycled Attention](https://arxiv.org/html/2411.05787v1), [IndexCache](https://github.com/THUDM/IndexCache), [HySparse](https://arxiv.org/html/2602.03560), [LatentIndex](https://arxiv.org/html/2610.04635)
- [Conf-KV](https://arxiv.org/html/2605.24786v1), [SinkRouter](https://arxiv.org/pdf/2604.16883), [QuantSpec](https://arxiv.org/pdf/2502.10424), [OasisKV](https://arxiv.org/pdf/2608.08097), [DualDecoder](https://arxiv.org/pdf/2607.26475)
- [TokenSelect](https://aclanthology.org/2025.emnlp-main.1079.pdf), [Prefix-aware attention](https://arxiv.org/pdf/2505.17694), [RazorAttention](https://arxiv.org/pdf/2407.15891)

**Background for the proposal:**

- [PagedAttention (vLLM)](https://ar5iv.labs.arxiv.org/html/2309.06180)
- [Mind the Memory Gap](https://arxiv.org/html/2503.08311v2)

A web search cannot prove that an idea is unpublished. Searches ran on 15 and 18 November 2026, and later work will be missing.
