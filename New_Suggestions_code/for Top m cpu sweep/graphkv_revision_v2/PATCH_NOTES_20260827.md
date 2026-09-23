# 2026-08-27 Colab compatibility patch

## Symptom

Preflight incorrectly reported that these vLLM flags were missing:

- `--gpu-memory-utilization`
- `--kv-transfer-config`
- `--no-enable-prefix-caching`
- `--pipeline-parallel-size`
- `--tensor-parallel-size`

## Cause and fix

Recent vLLM releases use paged command help. Plain
`vllm serve --help` omits several engine-configuration groups, while
`vllm serve --help=all` displays the complete option set. Both preflight and
the isolated runtime launcher now inspect complete help and retain a fallback
for older vLLM releases that only support plain help.

Preflight also saves the exact inspected help text. If a real mismatch remains,
it writes `preflight_cli_failure.json` with package versions, executable paths,
and version-command output.

## TorchAudio

GraphKV does not use TorchAudio. Do not install a nightly TorchAudio wheel to
repair this CLI-help error. If TorchAudio is already mismatched, leave it
uninstalled for this experiment:

```bash
uv pip uninstall --system torchaudio
```

Restart the notebook runtime only when PyTorch, vLLM, LMCache, or a compiled
CUDA extension was changed. Replacing this source bundle alone does not require
a runtime restart.
