# Phase 0: go/no-go measurement

Goal: find out, on a real model, whether tiered KV reads can cut the bytes read per decode step enough to matter without hurting quality. Budget: 1-2 weeks.

## Questions

1. **Concentration.** Under the tiering rule below, what fraction of blocks is HOT, WARM and COLD, per layer and head?
2. **Stability.** How many steps does a plan stay valid before it has to be recomputed?
3. **Quality.** What do tiered reads cost in output error and perplexity, compared with fp16, uniform 8-bit and uniform 4-bit?

## Setup

- **Debug model (Mac):** any 1-3B instruct model with grouped-query attention.
- **Real model (NVIDIA GPU):** Llama-3.1-8B-Instruct or an equivalent 8B model, fp16.
- **Data:** 20-50 prompts of 8K-32K tokens from LongBench and RULER, plus a needle-in-haystack set. Generate 128-256 tokens per prompt, greedy.
- **Block size:** 16 tokens, matching vLLM's default.

## Method

### A. Capture

At every decode step and layer, wrap the attention module's forward to get the query for the new token and the cached keys, and compute that token's attention weights over the context (n values per head). Do not ask the library for full attention maps.

Save, per step, layer and query head, two arrays over blocks: mass `m_b` and squared mass `q_b`. Keep every query head: rule B tiers each query head and then takes the highest tier per KV head, which needs the unreduced arrays. Reducing query heads to their KV head with a max before tiering is a separate variant to compare against rule B, computed at analysis time. It is not a storage shortcut. When storing in fp16, store `sqrt(q_b)` instead of `q_b`, because `q_b` underflows fp16 at long context, and record which one is stored. Subsample steps if storage becomes a problem, keeping runs of consecutive steps long enough for the stability measurement (R up to 16).

Check first that the chosen model caches keys after the rotary embedding is applied. The design assumes cached keys can be scored and reordered without position bookkeeping. Run the check at a late position and compare only dimensions that RoPE has rotated substantially there: low-frequency dimensions barely rotate, and in some models (Qwen2.5) large key biases sit in them, so a comparison over the whole key vector cannot tell rotated keys from unrotated ones.

### B. Tiering rule

For one head, with attention weights `w_i` over context tokens and blocks `b`:

- `m_b = sum of w_i in b`, `q_b = sum of w_i^2 in b`.
- **HOT:** sort blocks by `q_b` descending and promote until the `q_b` left outside HOT sums to at most `eps_w`.
- **COLD:** sort the remaining blocks by `m_b` ascending and mark them COLD while their cumulative mass stays below `eps_c`.
- **WARM:** everything else.
- Always HOT: block 0 and the last two blocks.
- With grouped-query attention, a KV head takes the highest tier any of its query heads asks for.
- Variant for comparison (max-reduced): first take, per block, the max of `m_b` and of `q_b` over a KV head's query heads, then apply the rule once per KV head. Report both.

Sweep `eps_w` over {1e-2, 1e-3, 1e-4} and `eps_c` over {0.5%, 2%, 5%}.

### C. Stability

Compute a plan at step t and apply it unchanged at steps t+1 to t+R, for R in {1, 2, 4, 8, 16}. Record:

- attention mass that lands in COLD blocks under the stale plan;
- Jaccard overlap between the stale and fresh HOT sets;
- missed blocks: HOT under a fresh plan but COLD under the stale one.

### D. Simulated tiered attention

Prefill normally at fp16. Then decode token by token with a patched attention forward that, for the new token, uses fp16 keys and values for HOT blocks, quantised keys and values for WARM blocks, and masks COLD blocks.

- **Quantisers:** 8-bit and 4-bit uniform, per channel for keys and per token for values. Use nearest rounding. Keys are quantised per channel within each 16-token block. The 1-bit key plane uses the same quantiser with two levels. Also report a mean-absolute-deviation 1-bit plane (sign around the per-channel mean), because min-max with two levels loses most of each key.
- **Variants:**
  - **a. Fresh plan** every step, from the exact weights of that step. An oracle: the quality ceiling for the rule.
  - **b. Stale plan**, R in {4, 8}, no check. Shows the damage plan reuse does on its own.
  - **c. Stale plan + same-step promotion.** Apply the tiering rule to this step's coarse scores: fp16 for HOT blocks, 4-bit keys for WARM blocks, the 1-bit key plane for COLD blocks. Any block it marks HOT that was not read at fp16 gets an fp16 read in a second pass.
  - **d. Two tiers** (no COLD, `eps_c = 0`) with the same same-step promotion.
  - **e. COLD keys at 4 bits.** As c, but COLD blocks have their 4-bit keys read every step (values skipped), so promotion scores them like WARM blocks. No separate 1-bit plane and no periodic refresh scan.
  - **Re-based ladder.** In c, d and e the tiers sit on the uniform ladder found by the pre-screen: HOT is read at b* (fp16 if b* = 16), WARM at the next width down, COLD as above. Fix-up reads and the HOT scores used for promotion are at b* too.
  - Plans in b-e are refreshed every R steps from that step's exact weights. Blocks written since the refresh are HOT.
- **Key codecs:** how key scales and offsets are shared, for uniform and tiered variants alike: per channel per 16-token block; per channel shared over 256-token groups (blocks of an incomplete group use per-block values until it completes); static per-layer, per-head, per-channel values calibrated on a few prompts kept apart from evaluation (out-of-range keys saturate). Values stay per token.
- **Baselines:** fp16; fp8 e4m3 (per-tensor scale 1, saturating), labelled **emulation only** until reproduced with vLLM's real fp8 KV cache on the cluster, and never a candidate for b*; uniform 8-, 4-, 3- and 2-bit everywhere, each with every key codec; HOT blocks only with the rest dropped. The newest, partial block is fp16 in the uniform and tiered variants.
- **Perplexity reference:** a held-out continuation of the prompt's own document (the next 128 tokens of a long LongBench document, teacher-forced), so every prompt contributes the same number of scored tokens. Report the pooled perplexity change with a 95% bootstrap interval over prompts.
- **Pre-screen (MPS, before cluster runs):** fp16, fp8, uniform 8-, 4-, 3- and 2-bit with each key codec, on about 40 prompts at 4K with at least 5K scored tokens, for a 1.5B and a 3B model. Its result fixes b* and the ladder (below) for the tiered runs.
- **Metrics:** relative L2 error of each layer's attention output; perplexity of a reference continuation under teacher forcing; token agreement with the fp16 greedy output; the benchmark's task score where one exists.
- **For c, d and e, also:** fix-up rate per step (blocks given a second-pass fp16 read); recall of the same-step scan on stale-COLD events (blocks HOT under the fresh plan but COLD under the stale one; for c this is the 1-bit scan) and its precision; rho including the scan and fix-up reads.

### E. Read fraction

`rho = HOT * 1.0 + WARM * (c/16) + COLD * 0.5 * (c/16) / R`

where the fractions are of blocks, `c` is the coarse-plane bit width, and the last term is the periodic scan of coarse keys in COLD blocks. `rho = 1` means reading everything at fp16.

For variants c and d, add the same-step reads, with PW and PC the fractions of blocks promoted from WARM and from COLD:

- c: `rho = HOT + (WARM - PW) * (c/16) + PW * (0.5 * c/16 + 1) + COLD * 0.5 * (1/16) + PC + COLD * 0.5 * (c/16) / R`
- d: `rho = HOT + (WARM - PW) * (c/16) + PW * (0.5 * c/16 + 1)`
- e: `rho = HOT + (WARM - PW) * (c/16) + PW * (0.5 * c/16 + 1) + COLD * 0.5 * (c/16) + PC`
- fp8: `rho = 0.5`

These count code bits only. Decisions use rho with quantiser metadata (`rho_meta`): an fp16 scale and offset per channel for keys, shared by 1 block, 16 blocks (256-token groups) or none (static), and per token for values. With head dim 128, a 4-bit block read costs 0.32 of an fp16 block with per-block key scales, 0.26 with 256-token groups and 0.26 static; a 1-bit key scan costs 0.09, 0.035 and 0.03.

## Outputs

- A table of rho (code bits and with metadata) against perplexity increase, with its bootstrap interval, for each configuration, including variants a-e, the baselines and each key codec.
- A per-layer, per-head heatmap of HOT, WARM and COLD fractions.
- Stability curves against R.

## Decision

"Within 1%" means the pooled teacher-forced perplexity increase over fp16 is at most 1% (point estimate), paired on the same prompts. Report its 95% bootstrap interval over prompts and flag any deciding row whose upper bound is above 1%. Bytes always include quantiser metadata (`rho_meta`).

- **Bar:** the cheapest uniform cache that stays within 1%, across all key codecs and widths (fp16, uniform 8/4/3/2-bit), bytes including metadata. fp8 is emulation only and is not a candidate. On the 1.5B pre-screen this is group256 4-bit (rho_meta 0.264), so the budget is rho_meta <= 0.198.
- **Ladders:** HOT at the bar's width, WARM at a lower width (4/3 and 4/2 with group256 keys), COLD as now. Per-codec b\* (lowest uniform width within 1%) is reported to guide ladder choice.
- **Go** if a tiered variant (c, d or e) stays within 1% and reads at least 25% fewer bytes than the bar.
- **No-go** otherwise. Flag a non-monotone uniform result.
- **If every tiered variant fails on quality:** run variant a (fresh plan) on the 4/2 ladder to separate "ladder too coarse" from "stale plan", then try per-block keys on a 4/3 ladder as the fallback codec.

Variants a and b do not qualify: a needs exact weights to plan, and b has no check against a stale plan. Report R, the effective HOT fraction (promoted blocks counted as HOT) and the stale-COLD tail beside every deciding row: the fraction of query heads with more than 2%, 5% and 10% of their attention mass in COLD blocks, before and after promotion (`frac_qheads_cold_mass_gt_*`), plus the stability-analysis tail from part C. `analyze_d.py` applies this rule and writes `decision.json`.

## Known pitfalls

- A 16K-token prefill with a materialised attention matrix needs about 17 GB per layer in fp16. On MPS, prefill in chunks with the cache if memory is tight.
- Tiered attention has to be applied step by step. A single teacher-forced pass over the continuation would compute every position's attention at once and bypass the plan.
- The 4-bit quantiser here is deliberately simple. If results are close to the threshold, repeat with a better codec before deciding.
