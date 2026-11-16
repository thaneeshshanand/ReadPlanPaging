# Read-plan paging (KV-cache research prototype)

## What this is

A research prototype testing one hypothesis: LLM serving throughput improves if each decode step reads only part of the KV cache. For every 16-token block, a sequence either skips it (COLD), reads a low-precision copy (WARM), or reads it at fp16 (HOT). The plan is stored per sequence beside vLLM's block table and reused across steps.

- Full proposal, prior art and risks: `docs/read-plan-paging-kv-cache.md`. Read it before any design work.
- Current task and exact definitions: `docs/PHASE0.md`.

## Status

Phase 0, the go/no-go measurement. Nothing has been measured on a real model yet. The only existing numbers are a bandwidth estimate and a synthetic-data check, both in the proposal.

## Environment

- **Local:** MacBook M2 Pro, 32 GB, PyTorch on MPS. Debugging only: a 1-3B model, prompts up to 4K tokens.
- **Real runs:** an NVIDIA GPU (Northeastern Explorer cluster through Slurm, or a rented 24 GB card). 8B-class model in fp16, prompts of 8K-32K tokens.
- Every script must run unchanged on `mps` and `cuda`: choose the device at runtime and take model name, context length and output directory from CLI arguments.

## Working rules

- Report only numbers produced by a run in this repo. Label anything else as an estimate.
- Each run writes its config, seed, model revision and git commit next to its results.
- Never materialise a full n x n attention matrix for long prompts. At 16K tokens that is about 17 GB per layer in fp16. Compute scores for the newest query against the cached keys instead.
- Keep capture and analysis separate: capture writes statistics to disk, analysis reads them.
- Ask before downloading a model over 10 GB or starting a job expected to run longer than an hour.
