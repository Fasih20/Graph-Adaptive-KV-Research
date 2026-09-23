"""Fresh-process, single-GPU vLLM + LMCache MP runtime.

Every policy/K/repetition receives a new runtime.  This is intentionally
slower than reusing one server: process isolation is the mechanism that
prevents native GPU prefix blocks from leaking between experimental arms.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from runtime_env import child_environment, collect_cli_help


class RuntimeLaunchError(RuntimeError):
    pass


def _wait_http(url: str, process: subprocess.Popen, log_path: Path, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = ""
            if log_path.exists():
                tail = "\n" + "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
            raise RuntimeLaunchError(
                f"process exited with code {process.returncode} before {url} became healthy:{tail}"
            )
        try:
            response = requests.get(url, timeout=2)
            if response.ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeLaunchError(f"timed out after {timeout_s:.0f}s waiting for {url}; see {log_path}")


def _flag_supported(help_text: str, flag: str) -> bool:
    return flag in help_text


@dataclass(frozen=True)
class RuntimeConfig:
    model: str
    model_revision: str | None
    gpu_id: str
    vllm_host: str
    vllm_port: int
    lmcache_host: str
    lmcache_port: int
    lmcache_http_port: int
    lmcache_prometheus_port: int
    max_model_len: int
    block_size: int
    chunk_size: int
    gpu_memory_utilization: float
    l1_size_gb: float
    startup_timeout_s: float
    seed: int
    blend_separator: str = " # # "
    trust_remote_code: bool = False
    max_num_seqs: int = 1


class IsolatedRuntime:
    def __init__(self, config: RuntimeConfig, run_dir: Path):
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.processes: list[tuple[str, subprocess.Popen, object]] = []
        self.commands: dict[str, list[str]] = {}

    def _lmcache_command(self, help_text: str, instance_id: str) -> list[str]:
        c = self.config
        required = [
            "--host", "--port", "--http-port", "--chunk-size", "--l1-size-gb",
            "--eviction-policy", "--engine-type", "--supported-transfer-mode",
        ]
        missing = [flag for flag in required if not _flag_supported(help_text, flag)]
        if missing:
            raise RuntimeLaunchError(f"installed lmcache CLI is missing required flags: {missing}")
        command = [
            "lmcache", "server",
            "--host", c.lmcache_host,
            "--port", str(c.lmcache_port),
            "--http-port", str(c.lmcache_http_port),
            "--chunk-size", str(c.chunk_size),
            "--l1-size-gb", str(c.l1_size_gb),
            "--eviction-policy", "LRU",
            "--engine-type", "default",
            "--supported-transfer-mode", "auto",
        ]
        if _flag_supported(help_text, "--instance-id"):
            command += ["--instance-id", instance_id]
        if _flag_supported(help_text, "--prometheus-port"):
            command += ["--prometheus-port", str(c.lmcache_prometheus_port)]
        if _flag_supported(help_text, "--metrics-sample-rate"):
            command += ["--metrics-sample-rate", "1.0"]
        if _flag_supported(help_text, "--enable-extra-logging"):
            command += ["--enable-extra-logging"]
        return command

    def _vllm_command(self, help_text: str) -> list[str]:
        c = self.config
        required = [
            "--kv-transfer-config", "--tensor-parallel-size", "--pipeline-parallel-size",
            "--gpu-memory-utilization", "--no-enable-prefix-caching",
        ]
        missing = [flag for flag in required if not _flag_supported(help_text, flag)]
        if missing:
            raise RuntimeLaunchError(f"installed vLLM CLI is missing required flags: {missing}")
        connector = {
            "kv_connector": "LMCacheMPConnector",
            "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": c.lmcache_host,
                "lmcache.mp.port": c.lmcache_port,
                "lmcache.mp.eager_prefetch": False,
                "lmcache.mp.mp_transfer_mode": "auto",
            },
        }
        command = [
            "vllm", "serve", c.model,
            "--host", c.vllm_host,
            "--port", str(c.vllm_port),
            "--max-model-len", str(c.max_model_len),
            "--block-size", str(c.block_size),
            "--tensor-parallel-size", "1",
            "--pipeline-parallel-size", "1",
            "--gpu-memory-utilization", str(c.gpu_memory_utilization),
            "--no-enable-prefix-caching",
            "--kv-transfer-config", json.dumps(connector, separators=(",", ":")),
            "--seed", str(c.seed),
        ]
        optional_switches = [
            "--enforce-eager",
            "--no-enable-chunked-prefill",
            "--no-async-scheduling",
            "--disable-hybrid-kv-cache-manager",
        ]
        command += [flag for flag in optional_switches if _flag_supported(help_text, flag)]
        if _flag_supported(help_text, "--max-num-seqs"):
            command += ["--max-num-seqs", str(c.max_num_seqs)]
        if c.model_revision and _flag_supported(help_text, "--revision"):
            command += ["--revision", c.model_revision]
        if c.trust_remote_code and _flag_supported(help_text, "--trust-remote-code"):
            command += ["--trust-remote-code"]
        return command

    def _spawn(self, name: str, command: list[str], env: dict[str, str]) -> tuple[subprocess.Popen, Path]:
        log_path = self.run_dir / f"{name}.log"
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self.processes.append((name, process, handle))
        self.commands[name] = command
        return process, log_path

    def start(self, instance_id: str) -> "IsolatedRuntime":
        c = self.config
        # Never mistake a stale server's /health response for this process.
        ports = [(c.vllm_host,c.vllm_port),(c.lmcache_host,c.lmcache_port),
                 (c.lmcache_host,c.lmcache_http_port),(c.lmcache_host,c.lmcache_prometheus_port)]
        if len(set(ports)) != len(ports):
            raise RuntimeLaunchError("Runtime ports must be distinct")
        for host, port in ports:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try: probe.bind((host, port))
                except OSError as exc:
                    raise RuntimeLaunchError(f"Port {host}:{port} is in use; stop the owning process explicitly") from exc
        if c.chunk_size % c.block_size:
            raise RuntimeLaunchError("LMCache chunk_size must be a multiple of vLLM block_size")
        env = child_environment(c.gpu_id)
        env.update({
            "LMCACHE_ENABLE_BLENDING": "False",
            "LMCACHE_CHUNK_SIZE": str(c.chunk_size),
            "LMCACHE_LOG_LEVEL": "INFO",
            "DO_NOT_TRACK": "1",
        })
        lm_help = collect_cli_help(["lmcache", "server"])
        vllm_help = collect_cli_help(["vllm", "serve"], exhaustive=True)
        lmcache = self._lmcache_command(lm_help, instance_id)
        vllm = self._vllm_command(vllm_help)
        lm_proc, lm_log = self._spawn("lmcache", lmcache, env)
        _wait_http(
            f"http://{c.lmcache_host}:{c.lmcache_http_port}/healthcheck",
            lm_proc,
            lm_log,
            min(120, c.startup_timeout_s),
        )
        vllm_proc, vllm_log = self._spawn("vllm", vllm, env)
        _wait_http(
            f"http://{c.vllm_host}:{c.vllm_port}/health",
            vllm_proc,
            vllm_log,
            c.startup_timeout_s,
        )
        (self.run_dir / "launch_commands.json").write_text(
            json.dumps(self.commands, indent=2), encoding="utf-8"
        )
        return self

    def stop(self) -> None:
        for _, process, _ in reversed(self.processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 30
        for _, process, _ in reversed(self.processes):
            remaining = max(0.1, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
        for _, _, handle in self.processes:
            handle.close()
        self.processes.clear()

    def __enter__(self) -> "IsolatedRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
