# Read-plan paging: a read-optimized successor to PagedAttention

*Research proposal and project idea. Written 15 Nov 2026. "Read-plan paging" is a working name.*

**Status.** The hardware facts and prior-art claims below are sourced. The design is a hypothesis: nothing here has been run on a real model or a GPU. The only numbers of my own are a bandwidth-only roofline estimate and a synthetic-data sanity check, both labelled as such.

## 1. Thesis

PagedAttention fixed how KV-cache memory is *allocated*. The limit that serving engines hit now is how many KV bytes are *read from HBM per generated token*. vLLM still makes one precision decision per deployment (fp16, fp8 or NVFP4) and reads every block of every sequence in full on every step.

The proposal: keep vLLM's blocks and block table exactly as they are, and give the block table a second job. Besides *where* a block lives, each sequence records *how much of that block it needs to read right now*: skip it, read a cheap low-precision plane, or read it in full. The plan is derived from attention mass the kernel already computes, reused across steps, and re-checked by a cheap scan.

Yardstick for everything below: **HBM bytes read per generated token**, at equal output quality.

## 2. What the GPU is doing during decode

| Fact | Value | Source |
|---|---|---|
| H100 SXM HBM bandwidth / dense fp16 compute | 3.35 TB/s / 989 TFLOPS, so roughly 295 ops per byte available | [Spheron](https://www.spheron.network/blog/nvidia-h100-specs/), [Baseten](https://www.baseten.co/inference-engineering/book/02-models/calculating-inference-bottlenecks/) |
| Decode attention needs | about 0.5 to 1 op per byte, flat as batch grows | [Mind the Memory Gap](https://arxiv.org/html/2503.08311v2) |
| Attention kernel at max batch | over half of warp cycles idle waiting on memory | same |
| Throughput plateau | OPT-1.3B reaches near-max throughput using about 40% of its KV memory | same |
| Root cause | KV data is streamed with little temporal reuse; compute sits underused | [SLIM](https://arxiv.org/pdf/2607.29575) |
| On-chip cache | 50 MB of L2 in front of 80 GB of HBM | [NVIDIA](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) |

My own arithmetic for a Llama-3.1-8B-class model (32 layers, 8 KV heads, head dim 128, fp16): 128 KiB of KV per token, 4 FLOP per KV byte with GQA (1 without, which matches the measured range above). At batch 24 with 16K contexts the KV cache is 51.5 GB and makes up 76% of the bytes streamed per step; weights are the other 24%.

Two consequences shape the design:

1. Compute is nearly free during decode and bandwidth is not, so spending FLOPs (dequantisation, an extra scoring pass) to avoid reading bytes is a good trade.
2. Once throughput plateaus, HBM capacity is no longer the binding constraint. PagedAttention's objective (fewest bytes allocated) stops being the right one; spare memory can be spent on read-optimized representations.

## 3. What PagedAttention did and did not fix

Fixed: fragmentation and over-reservation, through fixed-size blocks, a logical-to-physical block table, on-demand allocation and copy-on-write sharing, for a reported 2 to 4x throughput gain ([paper](https://ar5iv.labs.arxiv.org/html/2309.06180)).

Not fixed:

- **Sharing saves memory, not reads.** A shared prefix block is still read once per sequence per step ([Hydragen](https://arxiv.org/html/2402.05099v2)). PAT measured KV loads at 4.3 to 8.7x the theoretical minimum on shared-prefix traces ([PAT](https://arxiv.org/pdf/2511.22333)).
- **Every block is read at full stored precision on every step**, whatever attention weight it ends up getting.
- **Precision is chosen at store time**, before any future query is known.

## 4. Prior art, so the novelty claim is honest

I generated two "new" ideas before this one. Both turned out to be published, one of them five weeks ago. They are listed here with everything else adjacent.

| Direction | Work | What it does | What it leaves open |
|---|---|---|---|
| Contiguity via GPU virtual memory | [vAttention](https://www.microsoft.com/en-us/research/publication/vattention-dynamic-memory-management-for-serving-llms-without-pagedattention/) (ASPLOS 25) | Contiguous virtual KV, demand-paged physical memory; up to 1.23x over paged kernels | Still reads everything |
| Read shared prefix once | [Hydragen](https://arxiv.org/html/2402.05099v2), [FlashInfer](https://arxiv.org/pdf/2501.01005), [PAT](https://arxiv.org/pdf/2511.22333) | Batch queries over shared KV; PAT is a vLLM plugin | Only helps shared prefixes |
| Page-level selection | [Quest](https://hanlab.mit.edu/projects/quest) (ICML 24) | Per-page min/max key metadata, read top-K pages | Fixed budget, misses possible, weak on diffuse heads |
| Cluster index + centroid stand-ins | [ClusterKV](https://arxiv.org/html/2412.03213v2), [RetroInfer](https://arxiv.org/pdf/2505.02922), [Multipole Attention](https://arxiv.org/pdf/2506.13059), [LycheeCluster](https://arxiv.org/abs/2603.08453) | Cluster keys, read important clusters exactly, approximate the rest by centroids; 2 to 4.5x decode speedups reported | Index build and maintenance; not tied to the paged allocator or prefix sharing |
| Error guarantees | [vAttention: Verified Sparse Attention](https://arxiv.org/html/2510.05688v1) (ICLR 26) | Top-k plus sampling with (epsilon, delta) guarantees | Algorithm-level, not a serving memory design |
| Store-time mixed precision | [DiffKV](https://arxiv.org/html/2412.03131v3) (SOSP 25), fp8 / NVFP4 KV in [vLLM](https://docs.vllm.ai/en/stable/api/vllm/v1/attention/backends/flashinfer/) | Precision per token fixed when stored; DiffKV needs a new on-GPU allocator for the irregular memory | Irreversible. Token eviction also frees nothing in vLLM until a whole block is empty ([NVIDIA](https://research.nvidia.com/labs/eai/blogs/kv-cache-compression-and-its-infra-problems/)) |
| Low-bit scan to find what matters | [SpeCache](https://arxiv.org/html/2503.16163v1), [FIER](https://arxiv.org/pdf/2508.08256), [SparQ](https://arxiv.org/html/2312.04985v6) | 1 to 2 bit copy or partial channels to locate top-k, then fetch full precision | Rest of the context is dropped or read at one fixed precision |
| **Read-time precision** | [ReadKV](https://arxiv.org/html/2610.11245v1) (Oct 2026) | Nested code; each query chooses how many bits to read per key channel and per value token | Key depth is set per channel, not per token; its GPU reader reads at least one bit of everything; allocation is recomputed for every query; tested on one layer at batch 1, where dense was faster; batched serving and prefix sharing not evaluated |
| Per-layer regime choice | [TailorKV](https://arxiv.org/pdf/2505.19586) | Dense-attention layers tolerate heavy quantisation, sparse ones want top-k retrieval; chosen offline per layer | Static, per layer |

The gap I could find: read-time decisions exist as algorithms, but none is expressed in the paged allocator's own data structure, none amortises the decision across steps, and the newest one (ReadKV) has not been shown to win in batched serving. Its authors say so and name that as the direct test.

## 5. The design

### 5.1 Layout

- vLLM's block pool and block table are unchanged. Blocks stay immutable once full, so prefix caching and copy-on-write keep working.
- A second **coarse pool**, indexed by the same physical block ids, holds a low-precision encoding of each block.
  - v0: an fp8 or NVFP4 copy, using KV dtypes vLLM's FlashInfer backend already supports. Costs 25% extra memory at 4 bits, which is affordable at the bandwidth plateau.
  - v1: nested bit planes in the style of ReadKV, so the coarse plane is a prefix of the full code and nothing is duplicated.
- A **read plan** sits beside each sequence's block table: 2 bits per block per layer per KV head, meaning COLD, WARM or HOT. For a 16K context that is 1,024 blocks x 32 layers x 8 heads x 2 bits = 64 KB per sequence.

The plan belongs to the sequence, not to the block. Two requests sharing a system-prompt block can read it at different tiers without touching the block. That is the property store-time compression lacks.

### 5.2 One decode step, per layer

1. Tail blocks written since the last refresh, plus the first block (attention sink), are HOT.
2. HOT blocks: read from the fp16 pool, producing a partial attention state (output, log-sum-exp).
3. WARM blocks: read from the coarse pool, producing a second partial state.
4. Merge the two states exactly with the log-sum-exp merge that cascade/prefix attention already uses (vLLM ships `merge_attn_states`).
5. COLD blocks are not read.
6. Both passes emit per-block attention mass as a by-product. These are the "reference bits" of the paging analogy and feed an exponential moving average per block.

Every R steps, staggered across sequences so the cost is smooth, a **refresh** scans the coarse keys of COLD blocks and recomputes tiers. A WARM block whose mass crosses the hot threshold is promoted for the next step without waiting.

### 5.3 Tiering rule

Leaving a block at coarse precision adds error variance roughly in proportion to the squared attention weights inside it. So:

- Promote blocks to HOT in descending order of summed squared weight until the squared weight still at coarse precision is below a budget eps_w.
- Demote to COLD the lowest-mass blocks that together hold less than eps_c of the mass.
- With GQA, a KV head's tier is the highest tier any of its query heads asks for.

This single rule produces the right behaviour per head without an offline classifier. A diffuse head has no block worth promoting, so everything is WARM and it behaves like a quantised cache. A sharp head gets a few HOT blocks and a large COLD set, so it behaves like sparse retrieval. That is TailorKV's layer-level observation, applied per head and per step. Using squared attention as the importance signal matches what ReadKV uses for values.

My first rule (promote until X% of mass is covered) was wrong: the synthetic check showed it marks three quarters of a diffuse head HOT. The squared-weight rule fixed it.

### 5.4 Why it suits the hardware

- Reads stay block-contiguous. The unit of selection is the allocator's own 16-token block, so there is no per-token gather.
- Dequantisation and the second pass use FLOPs that decode leaves idle.
- v0 should need no new CUDA: two FlashInfer paged-decode calls with filtered page lists (the block-sparse format already expresses per-query page subsets) plus the existing merge. Confirm against the current FlashInfer API before relying on this.
- Coarse planes are small: one layer's 4-bit key plane for about 98K active tokens fits in 50 MB. Hopper's L2 prefetch instruction, which an [async-prefetch kernel](https://arxiv.org/pdf/2504.06319) already uses for ordinary blocks, could pull coarse blocks ahead of use. Unverified; treat as a stretch item.

### 5.5 New versus borrowed

Borrowed: read-time precision (ReadKV), low-bit scans to locate important tokens (FIER, SpeCache), page-granular selection (Quest), the dense/sparse regime split (TailorKV), state merging (Hydragen, FlashInfer).

New, as far as my search found:

1. The read plan lives in the per-sequence block table over immutable shared blocks, so read-time tiering composes with prefix caching.
2. Three tiers per block under one error-budget rule, which unifies quantisation-style and sparsity-style reads per head and per step.
3. Plan reuse across steps with a refresh cadence, which amortises the selection cost that made ReadKV slower than dense in its single-layer test.
4. An end-to-end test in batched, paged serving.

This area moves weekly. Re-run the literature check before claiming novelty in public.

## 6. Expected gains

### 6.1 Roofline estimate (bandwidth only, not a measurement)

Assumes a step streams all weights plus a fraction rho of active KV bytes at 3.35 TB/s. Ignores CPU overhead, kernel launch cost and imperfect bandwidth use, so read the ratios, not the absolute numbers.

| Operating point (Llama-3.1-8B class, H100) | fp16 KV | fp8 KV | rho 0.27 | rho 0.22 | rho 0.15 |
|---|---|---|---|---|---|
| Batch 32, 4K context (KV = 52% of traffic) | 3,225 tok/s | 4,349 | 5,179 (1.61x fp16, 1.19x fp8) | 5,403 (1.68x, 1.24x) | 5,752 (1.78x, 1.32x) |
| Batch 24, 16K context (KV = 76% of traffic) | 1,189 tok/s | 1,922 | 2,682 (2.26x, 1.40x) | 2,934 (2.47x, 1.53x) | 3,379 (2.84x, 1.76x) |

rho for a few tier mixes with a 4-bit coarse plane: 5% HOT, 30% WARM, refresh every 8 steps gives 0.135; 10% HOT, 50% WARM, refresh every 4 gives 0.24; 10% HOT with everything else WARM and no COLD gives 0.33.

Short contexts gain little because weights dominate the traffic. The scheme should switch on only above a context-length threshold.

### 6.2 Synthetic sanity check (not evidence about real models)

Random keys and values, 8,192 tokens, head dim 128, 16-token blocks, naive min-max uniform quantisers, 8 seeds. Relative error of the attention output against fp32; traffic as a fraction of reading everything at fp16.

| Head type | Uniform 8-bit (traffic 0.50) | Uniform 4-bit (0.25) | Tiered, 4-bit coarse, eps_w = 1e-3 | Same HOT set only, rest dropped |
|---|---|---|---|---|
| Sharp (top 1% of tokens hold 94% of mass) | 0.8% | 13.5% | 0.9% at 0.22 to 0.24 | 10.0% |
| Heavy-tailed (53%) | 0.4% | 7.5% | 2.8% at 0.26 to 0.27 | 39% |
| Diffuse (5%) | 0.03% | 0.5% | 0.5% at 0.25 | fails |

What this does and does not show. It shows the arithmetic works as intended: at 4-bit traffic the tiered read recovers nearly all of the gap to 8-bit on the sharp head and about two thirds of it on the heavy-tailed one, costs nothing beyond uniform 4-bit on diffuse heads, and beats block top-k with the same HOT set in every regime. It does not show how concentrated real attention is per block, how large the COLD set can be, or how per-layer errors compound through a model. Real codecs are also better than min-max at 4 bits, so the gap to uniform 4-bit will be smaller in practice. Stochastic rounding was worse than nearest rounding, so I dropped it.

## 7. Risks and kill criteria

1. **Important tokens are scattered.** FIER reports that pages mix important and unimportant tokens. If so, the HOT fraction in blocks inflates. Kill if HOT blocks exceed about 20% at the target quality.
2. **Plan staleness.** A token in a COLD block can become important between refreshes. Measure hit rate against R. Fallback: scan a 1-bit key plane of COLD blocks every step (FIER finds 1-bit keys largely preserve the top-k set), at about 3% extra traffic.
3. **Error compounding** across layers and long generations. Needs perplexity, LongBench, RULER and needle tests, not single-layer error.
4. **Overhead.** Two passes plus a merge and plan upkeep. Plan logic must stay on the GPU; CPU time already reaches about 30% of a step at batch 512 in the Memory Gap study.
5. **Memory.** v0 uses 1.25x the fp16 footprint and so loses when the deployment is memory-bound rather than bandwidth-bound. v1 is 1.0x. Moving 70% of refinement planes to host RAM would give about 0.47x.
6. **A fixed NVFP4 cache may be good enough.** If uniform 4-bit quality is acceptable for the target workload, the added machinery buys only the COLD savings.

## 8. Build plan

| Phase | Work | Hardware | Output |
|---|---|---|---|
| 0. Measure (1 to 2 weeks) | Hook attention in Hugging Face transformers on Llama-3.1-8B or Qwen with 8K to 32K prompts. Record block-mass concentration per head, tier fractions under the rule, plan stability across steps, and simulate tiered reads with real fp8 and 4-bit quantisers | One 24 GB GPU | rho versus perplexity curve. Go if rho <= 0.3 at <= 1% perplexity increase with R >= 4 |
| 1. Reference (2 to 3 weeks) | Custom cache class with dual pools and the plan; end-to-end generation | Same | LongBench, RULER, needle results against fp16, fp8, 4-bit |
| 2. Serving (3 to 4 weeks) | vLLM attention-backend plugin (the route PAT took): two FlashInfer paged-decode calls plus `merge_attn_states` | A100 or H100 rental | Tokens/s and time per output token over a batch x context grid; HBM bytes per token from profiler counters |
| 3. Stretch | Nested planes, fused kernel, coarse planes under shared-prefix batching, host-RAM demotion | | |

Phase 0 is cheap and decides the rest. A clean negative result there is still a useful measurement to share: block-level attention concentration and plan stability in current open models.

## 9. Ideas considered and set aside

- **Re-page by key similarity with multi-resolution centroid summaries** (a "mip-mapped" or Barnes-Hut KV cache). Published as Multipole Attention, RetroInfer and LycheeCluster.
- **Pure bit-plane read-time precision.** Published as ReadKV in October 2026.
- **Read shared prefixes once.** Published as Hydragen, FlashInfer composable formats and PAT.
- **Exact reuse across decode steps.** Each step's query depends on the previous step's output, so no KV read can be shared across steps without multi-token generation (speculative decoding already amortises this).
