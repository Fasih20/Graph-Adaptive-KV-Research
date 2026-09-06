"""Portable runtime selection, preflight validation, and run manifests.

The benchmark must never silently combine Kaggle's two T4s.  The selected
physical GPU is exposed to child processes as their only CUDA device and
vLLM is also launched with tensor parallel size one.  This module deliberately
uses subprocess/importlib metadata instead of importing torch so it can run
before any CUDA-aware package initializes.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REQUIRED_VLLM_FLAGS = {
    "--kv-transfer-config",
    "--tensor-parallel-size",
    "--pipeline-parallel-size",
    "--gpu-memory-utilization",
    "--no-enable-prefix-caching",
}
REQUIRED_LMCACHE_FLAGS = {
    "--host",
    "--port",
    "--http-port",
    "--l1-size-gb",
    "--eviction-policy",
    "--engine-type",
    "--chunk-size",
}


class PreflightError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_capture(args: list[str], timeout_s: float = 30.0) -> dict:
    try:
        proc = subprocess.run(
            args,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
        return {"command": args, "returncode": proc.returncode, "output": proc.stdout}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": args, "returncode": None, "output": str(exc)}


def query_gpus() -> list[dict]:
    result = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if result["returncode"] != 0:
        return []
    gpus = []
    for line in result["output"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "uuid": parts[2],
                "memory_total_mib": int(float(parts[3])),
                "driver_version": parts[4],
            }
        )
    return gpus


def resolve_gpu(gpu_id: str, gpus: list[dict]) -> dict:
    selected = [gpu for gpu in gpus if gpu["index"] == str(gpu_id) or gpu["uuid"] == str(gpu_id)]
    if len(selected) != 1:
        available = [f"{g['index']}:{g['name']}" for g in gpus]
        raise PreflightError(f"GPU {gpu_id!r} was not found; available GPUs: {available}")
    return selected[0]


def child_environment(gpu_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTHONHASHSEED", "0")
    return env


def port_is_available(host: str, port: int) -> bool:
    bind_host = "127.0.0.1" if host in {"localhost", "0.0.0.0"} else host
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((bind_host, int(port)))
        except OSError:
            return False
    return True


def collect_cli_help(command: list[str], *, exhaustive: bool = False) -> str:
    """Return usable CLI help across old and paged-help releases.

    Recent vLLM releases intentionally show only a small help page for
    ``vllm serve --help``.  Their complete engine arguments are exposed by
    ``--help=all``.  Older releases may reject that spelling and put every
    option in plain ``--help`` instead, so exhaustive inspection tries both
    forms and combines every successful response.
    """

    help_args = ("--help=all", "--help") if exhaustive else ("--help",)
    attempts: list[dict] = []
    successful: list[str] = []
    usage_fallbacks: list[str] = []

    for help_arg in help_args:
        result = run_capture([*command, help_arg], timeout_s=90)
        attempts.append(result)
        output = str(result.get("output") or "")
        if result["returncode"] == 0 and output.strip():
            successful.append(output)
        elif result["returncode"] == 2 and "usage:" in output.lower():
            # Some argparse CLIs use exit code 2 while still printing complete
            # usage.  Prefer a code-0 response, but retain this as a fallback.
            usage_fallbacks.append(output)

    texts = successful or usage_fallbacks
    if texts:
        return "\n\n".join(texts)

    details = "\n\n".join(
        f"$ {' '.join(item['command'])}\n"
        f"exit={item['returncode']}\n{str(item['output'])[-3000:]}"
        for item in attempts
    )
    raise PreflightError(f"Could not inspect {' '.join(command)} help:\n{details}")


def _missing_flags(help_text: str, required: Iterable[str]) -> list[str]:
    return sorted(flag for flag in required if flag not in help_text)


def source_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PreflightReport:
    created_at_utc: str
    platform: str
    python: str
    executable: str
    selected_gpu: dict
    visible_gpu_count_on_host: int
    child_cuda_visible_devices: str
    packages: dict
    commands: dict
    ports: dict
    source_hash: str
    passed: bool
    warnings: list[str]


def run_preflight(
    *,
    project_root: Path,
    output_dir: Path,
    gpu_id: str,
    vllm_host: str,
    vllm_port: int,
    lmcache_host: str,
    lmcache_port: int,
    lmcache_http_port: int,
    lmcache_prometheus_port: int | None = None,
    strict: bool = True,
) -> PreflightReport:
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    if platform.system() != "Linux":
        raise PreflightError("vLLM + LMCache benchmark requires Linux")
    if shutil.which("nvidia-smi") is None:
        raise PreflightError("nvidia-smi is not available; enable a GPU runtime")
    if shutil.which("vllm") is None:
        raise PreflightError("vllm executable is missing")
    if shutil.which("lmcache") is None:
        raise PreflightError("lmcache executable is missing")

    gpus = query_gpus()
    if not gpus:
        raise PreflightError("No NVIDIA GPU was reported by nvidia-smi")
    selected_gpu = resolve_gpu(gpu_id, gpus)
    if len(gpus) > 1:
        warnings.append(
            f"Host exposes {len(gpus)} GPUs; child processes will see only physical GPU {gpu_id}."
        )

    # vLLM 0.12+ uses paged help. Plain `--help` does not contain the engine
    # groups where the KV-transfer, parallelism, memory, and cache flags live.
    vllm_help = collect_cli_help(["vllm", "serve"], exhaustive=True)
    lmcache_help = collect_cli_help(["lmcache", "server"])
    (output_dir / "vllm_cli_help.txt").write_text(vllm_help, encoding="utf-8")
    (output_dir / "lmcache_cli_help.txt").write_text(lmcache_help, encoding="utf-8")
    missing_vllm = _missing_flags(vllm_help, REQUIRED_VLLM_FLAGS)
    missing_lmcache = _missing_flags(lmcache_help, REQUIRED_LMCACHE_FLAGS)
    if missing_vllm or missing_lmcache:
        diagnostics = {
            "python": sys.version,
            "python_executable": sys.executable,
            "vllm_executable": shutil.which("vllm"),
            "lmcache_executable": shutil.which("lmcache"),
            "vllm_package_version": package_version("vllm"),
            "lmcache_package_version": package_version("lmcache"),
            "vllm_version_command": run_capture(["vllm", "--version"]),
            "lmcache_version_command": run_capture(["lmcache", "--version"]),
            "missing_vllm_flags": missing_vllm,
            "missing_lmcache_flags": missing_lmcache,
        }
        (output_dir / "preflight_cli_failure.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )
        message = (
            f"Installed CLI is incompatible. Missing vLLM flags={missing_vllm}; "
            f"missing LMCache flags={missing_lmcache}. Diagnostic help and version "
            f"files were saved in {output_dir}."
        )
        if strict:
            raise PreflightError(message)
        warnings.append(message)

    ports = {
        "vllm": {"host": vllm_host, "port": int(vllm_port)},
        "lmcache_zmq": {"host": lmcache_host, "port": int(lmcache_port)},
        "lmcache_http": {"host": lmcache_host, "port": int(lmcache_http_port)},
    }
    if lmcache_prometheus_port is not None:
        ports["lmcache_prometheus"] = {
            "host": lmcache_host,
            "port": int(lmcache_prometheus_port),
        }
    busy = [name for name, value in ports.items() if not port_is_available(value["host"], value["port"])]
    if busy:
        raise PreflightError(f"Required ports are already in use: {busy}")

    packages = {
        name: package_version(name)
        for name in [
            "torch",
            "vllm",
            "lmcache",
            "transformers",
            "sentence-transformers",
            "numpy",
            "scipy",
            "pandas",
            "requests",
        ]
    }
    missing_packages = [name for name, version in packages.items() if version is None]
    if missing_packages:
        raise PreflightError(f"Required Python packages are missing: {missing_packages}")

    commands = {
        "nvidia_smi": run_capture(["nvidia-smi"], timeout_s=30),
        "pip_freeze": run_capture([sys.executable, "-m", "pip", "freeze"], timeout_s=120),
    }
    report = PreflightReport(
        created_at_utc=utc_now(),
        platform=platform.platform(),
        python=platform.python_version(),
        executable=sys.executable,
        selected_gpu=selected_gpu,
        visible_gpu_count_on_host=len(gpus),
        child_cuda_visible_devices=str(gpu_id),
        packages=packages,
        commands=commands,
        ports=ports,
        source_hash=source_tree_hash(project_root / "src"),
        passed=True,
        warnings=warnings,
    )
    (output_dir / "environment_manifest.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )
    return report


def stable_config_hash(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()
