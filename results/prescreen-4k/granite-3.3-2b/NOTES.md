# Granite 3.3 2B on MPS: stock SDPA decode is wrong in fp16

Found by tiered.py --verify on 2026-11-16 (prompt cont-gov_report-6de6104b947e-4096):
stock-SDPA teacher-forced ppl 290.0 vs patched fp16 path 1.458, while the per-step
attention-output check against SDPA passed (max rel err 4.8e-4).

Separate repro (model loaded with attn_implementation=sdpa / eager, no patch, same
prefill and teacher-forced decode over the 128 reference tokens):

| attention | dtype | narrativeqa-cb33b635d4a1 | gov_report-6de6104b947e |
|-----------|-------|--------------------------|-------------------------|
| sdpa      | fp16  | nan                      | nan                     |
| eager     | fp16  | 6.173                    | 1.458                   |
| sdpa      | fp32  | (not run)                | 1.458                   |

The patched path (fp32 attention for the new token) gives 6.176 and 1.458, matching
eager. Every variant in this directory decodes through the patched path; prefill uses
chunked stock SDPA and agrees with eager. Granite's attention scale is 1/64, not
1/sqrt(64); an fp16 overflow in the MPS decode kernel is a likely cause (not confirmed).

## Update 2026-11-17: not an fp16 overflow, and intermittent

Bisected with `tools/mps_decode_repro.py` (same prompt, stock sdpa, fp16):

- Plain stock decode is intermittent: NaN in 2 runs, correct (ppl 1.4585) in 2 later runs.
- Wrapping F.scaled_dot_product_attention with a read-only `torch.isfinite(out).all()`
  after each decode call gives NaN in every run (first at decode step 2). At the first
  bad call (layer 1) the new token's q and k already hold inf/NaN next to finite values
  near 6.3e4; v is finite. Wrapping with torch.mps.synchronize() instead is correct.
- With forward hooks on every module, the decode is correct and the largest activation
  anywhere is ~314, far below the fp16 range.

So the cause given above (fp16 overflow in the decode kernel) is not supported: the values
are garbage, not large, and read-only instrumentation decides whether they appear. Likely
an MPS race or buffer-reuse bug; not reduced below the full model. The conclusion for our
results is unchanged: every variant decodes through the patched path, which matched eager
and fp32 on all 40 prompts.

Full repro run (`mps_repro.log`, 2 runs per mode in one process, real prompt): check NaN in
1 of 2, plain correct 2 of 2, sync 2 of 2, eager 2 of 2 (1.4578), fp32 correct in run 0
but ppl 4.0842 (finite) in run 1. So the failure is not fp16-specific and can be silent
(finite but wrong). Our fp32-attention patched path is not proven immune by construction;
the evidence for our results is that it matched eager and fp32 on all 40 prompts under
--verify, and that e/f/uniform perplexities agree across stages and prompts. Worth
re-checking a few prompts on CUDA during the cluster verify stage.
