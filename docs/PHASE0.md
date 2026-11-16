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

- **Quantisers:** 8-bit and 4-bit uniform, per channel for keys and per token for values. Use nearest rounding.
- **Baselines:** fp16; uniform 8-bit everywhere; uniform 4-bit everywhere; HOT blocks only with the rest dropped.
- **Metrics:** relative L2 error of each layer's attention output; perplexity of a reference continuation under teacher forcing; token agreement with the fp16 greedy output; the benchmark's task score where one exists.

### E. Read fraction

`rho = HOT * 1.0 + WARM * (c/16) + COLD * 0.5 * (c/16) / R`

where the fractions are of blocks, `c` is the coarse-plane bit width, and the last term is the periodic scan of coarse keys in COLD blocks. `rho = 1` means reading everything at fp16.

## Outputs

- A table of rho against perplexity increase for each configuration.
- A per-layer, per-head heatmap of HOT, WARM and COLD fractions.
- Stability curves against R.

## Decision

**Go** if at least one configuration reaches all of:

- rho at or below 0.30;
- perplexity increase at or below 1%;
- R of 4 or more;
- HOT blocks at or below 20%.

Otherwise, write up the measurement and stop.

## Known pitfalls

- A 16K-token prefill with a materialised attention matrix needs about 17 GB per layer in fp16. On MPS, prefill in chunks with the cache if memory is tight.
- Tiered attention has to be applied step by step. A single teacher-forced pass over the continuation would compute every position's attention at once and bypass the plan.
- The 4-bit quantiser here is deliberately simple. If results are close to the threshold, repeat with a better codec before deciding.
