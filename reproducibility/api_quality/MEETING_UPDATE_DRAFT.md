# Supervisor update: GraphKV evaluation status

## What is complete

I repaired the GraphKV policy and the vLLM/LMCache measurement protocol. The
system now preserves document boundaries, fits offline adaptive weights only
on a training split, applies online updates prequentially, randomises arm order,
resets cache state between arms, and reports both ready-cache latency and the
currently synchronous end-to-end prototype cost.

The completed Qwen2.5-1.5B screening run used 200 unique events repeated twice
(400 paired observations per arm). At K=10:

| Policy | Prediction recall | Ready-cache TTFT |
|---|---:|---:|
| Cosine | 77.0% | 56.423 ms |
| Fixed graph | 86.0% | 53.500 ms |
| Online adaptive | 94.5% | 51.764 ms |

Against cosine, fixed graph improved prediction recall by 9.0 percentage
points (cluster-bootstrap 95% CI 1.50 to 16.67) and reduced ready-cache TTFT by
2.92 ms (CI -5.47 to -0.60). Online adaptive improved recall by 17.5 points
(CI 8.37 to 28.28) and reduced ready-cache TTFT by 4.66 ms (CI -8.35 to
-1.20). These are promising real-hardware screening results.

The important limitation is that candidate population is still synchronous in
the prototype. Therefore end-to-end prefetch time remains much higher than the
no-prefetch arm. The ready-cache result estimates the benefit once speculative
population overlaps useful work; it is not yet a claim of lower application
end-to-end latency.

A larger Qwen2.5-1.5B run is currently in progress. I will treat its output as
confirmatory only after the protocol report passes and all planned arms are
complete.

## What the older results contribute

- The old six-model simulation is useful as prediction/cost sensitivity, but
  not as six independent model validations because the causal model weights
  were not executed and several model curves are identical.
- The old local-model and Gemini/Groq quality experiments are exploratory. They
  show a mixed result: improving target-chunk retrieval does not automatically
  improve answer EM/F1.
- One positive old result is Qwen2.5-1.5B Hotpot at K=10, where adaptive minus
  cosine F1 was +0.0466 with a paired bootstrap interval of +0.0069 to +0.0865.
  Other settings were negative or inconclusive, so this is not a general claim.

## Quality-evaluation repair

I separated answer quality from systems latency and rebuilt the API evaluator.
It now uses HotpotQA's original document and supporting-sentence structure,
learns offline weights without API calls, updates online only after selecting
the current question's context, enforces equal reference-token budgets, and
reports paired confidence intervals for EM and token F1. It checkpoints each
policy call and no longer imposes the old fixed 5-RPM delay.

The next economical test is one stable Gemini Flash-Lite model, 60 held-out
questions, and K=10: at most 240 generation calls, often fewer when two policies
produce identical prompts. This can run on CPU and does not consume GPU time.

## Claims I am not making yet

- No universal improvement across all models or datasets.
- No absolute end-to-end latency win over no-prefetch while population is
  synchronous.
- No measured FLOP reduction; the valid hardware quantities are TTFT, request
  time, cache-token counters, traffic, and retrieval outcomes.
- No use of incomplete legacy rows as headline evidence.
