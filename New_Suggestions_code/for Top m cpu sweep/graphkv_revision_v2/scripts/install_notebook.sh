#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m pip install --upgrade uv
uv pip install --system -r requirements-gpu.txt -r requirements-analysis.txt

python - <<'PY'
import importlib.metadata as metadata
for name in ("torch", "vllm", "lmcache", "transformers", "sentence-transformers"):
    print(f"{name}=={metadata.version(name)}")
import lmcache.cuda_ops
print("LMCache CUDA extension import: OK")
PY

echo "Installation complete. Restart the Kaggle/Colab runtime before running the benchmark."

