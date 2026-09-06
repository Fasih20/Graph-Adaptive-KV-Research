# Audit of the supplied legacy GraphKV results

## What can be used

1. **Repaired real-vLLM/LMCache results:** primary systems evidence. The
   completed Qwen2.5-1.5B screen already shows a promising K=10 result: online
   adaptive improved prediction recall by 17.5 percentage points over cosine
   and reduced ready-cache TTFT by about 4.66 ms, with paired confidence
   intervals excluding zero. End-to-end time still loses to no-prefetch because
   the prototype populates candidates synchronously.
2. **Local causal-model quality CSVs:** exploratory answer-quality evidence.
   They contain real generated answers and question-level rows, so paired
   reanalysis is possible. They used the old selector, character chunks,
   hard-coded adaptive weights, and did not record equal token budgets; they do
   not validate the repaired adaptive method.
3. **Old Gemini/Groq aggregates:** exploratory evidence only. Only aggregate
   tables were supplied, so paired confidence intervals cannot be recovered.
4. **Six-model simulator:** legacy prediction and analytical-cost sensitivity,
   not six independent LLM validations. Qwen 1.5B/3B/7B and Gemma 1B produce
   essentially identical hit-rate curves because the causal model weights were
   not executed.
5. **Old vLLM folder:** exclude from claims; retain only for provenance.

## Reanalysis of the local-model quality rows

Paired bootstrap intervals expose a mixed result, not a universal graph win.

- Qwen2.5-1.5B Hotpot, K=10: old adaptive minus cosine F1 = **+0.0466**,
  95% bootstrap CI **[+0.0069, +0.0865]**.
- Qwen2.5-1.5B Hotpot, K=8: fixed graph minus cosine F1 = **-0.0465**,
  CI **[-0.0889, -0.0047]**.
- Qwen2.5-3B Hotpot, K=10: adaptive minus cosine F1 = **+0.0289**,
  CI **[-0.0034, +0.0626]**; inconclusive.
- Llama3.2-1B MuSiQue, K=6: adaptive minus cosine F1 = **-0.0519**,
  CI **[-0.0976, -0.0082]**.
- MultiFieldQA used only 150 rows despite an `n200` filename; its observed
  graph/adaptive differences have intervals crossing zero.

The sharp answer-quality collapse at large K is real in the stored outputs,
but token counts were not saved, so its cause cannot be established from these
files alone. Treat truncation/context dilution as hypotheses, not findings.

## Old API aggregate observations

- Hotpot and MuSiQue mostly favour cosine.
- Fixed graph is slightly ahead at 2Wiki Top-M=12 and MultiField Top-M=12.
- Adaptive is generally weakest.
- 2Wiki Top-M=20 completed 55/60 and MultiField Top-M=20 only 29/60; do not use
  those incomplete rows as headline comparisons.

## What to do now

- Let the current full vLLM run finish and preserve its entire output folder.
- Run the repaired API evaluator on CPU, beginning with the 3-question smoke
  test and then one 60-question, one-K held-out run.
- Do not rerun more “models” through the old simulator before the meeting.
- Present the old quality results as motivation for the repair: they show that
  retrieval-hit gains do not automatically guarantee answer-quality gains.
- Use the repaired evaluator's supporting-fact recall to separate retrieval
  failure from generator failure, and use its paired confidence intervals for
  claims.
