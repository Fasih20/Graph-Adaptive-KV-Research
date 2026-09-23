# Corrected implementation order

1. Dedicated cache-identity/numerical gate BEFORE asynchronous results.
2. Audit historical selection drift before claiming a small/no numerical change.
3. Shared ablation ladder: semantic-only, structure, two-hop, offline, online;
   explicit small M/K grids and fixed training/development/test roles.
4. Real bounded background requests with foreground and whole-cycle clocks.
5. Separate full-prompt local answer-quality evaluation, Hotpot/2Wiki/MuSiQue.
6. Atomic independent-block checkpoints and opt-in verified Drive snapshots.
7. GPU smoke, then choose a limited pilot matrix using measured runtime.

Not included: CacheBlend integration, cross-GPU producer/consumer transfer,
lost-in-the-middle, new paper claims, guarantees of significant improvements.

Correction to previous explanation: final packaged vLLM fixed and learned
policies already share the adaptive path scorer. Archived fixed/API scoring
differs. The prior guide's blanket claim should not be applied to final vLLM.

Gate answers are scoped: a pass supports exact-prefix reuse for the tested
model/runtime/prompts. It does not authorize independent-chunk KV concatenation.
