"""
vllm_launcher.py
=================
Process management for vLLM + LMCache CacheBlend.

Two paths, matching implementation_plan.md's "Integration Path Decision":

  MP MODE (default, recommended — no vLLM source patch):
    1. `lmcache server --engine-type blend ...` — standalone process, owns
       the KV store and the CacheBlend recompute logic.
    2. `vllm serve <model> ...` with `--kv-transfer-config
       {"kv_connector": "LMCacheMPConnector", ...}` — talks to (1) over the
       network, no source modification.

  IN-PROCESS MODE (fallback — 5-line vLLM patch, pinned install):
    Applies the exact edit from the blend_kv_v1 README (verbatim, see
    _INPROCESS_PATCH_* below) to the installed vllm package's
    v1/worker/gpu_worker.py, then launches a single `vllm serve` process
    with `LMCacheConnectorV1`.

CLI-flag disclaimer: the flags below are assembled from implementation_plan.md
§2/§5 plus the LMCache MP docs (docs.lmcache.ai/mp/index.html) and the
LMCache Kubernetes-operator docs' list of vLLM flags required by the
CacheBlend webhook (--attention-backend CUSTOM, --block-size 64,
--no-enable-chunked-prefill, --no-async-scheduling, --pipeline-parallel-size 1,
--enforce-eager). LMCache/vLLM's CLI surface moves fast — treat
`_build_vllm_mp_args` / `_build_lmcache_server_args` as the single place to
fix flag names if `smoke_test()` fails at startup rather than hunting
through the rest of the codebase.
"""

import os
import re
import sys
import time
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Optional, List

import requests

from exp_config import (MODEL_NAME, VLLM_HOST, VLLM_PORT, VLLM_BLOCK_SIZE,
                     LMCACHE_SERVER_HOST, LMCACHE_SERVER_PORT, LMCACHE_HTTP_PORT,
                     LMCACHE_ENV, ENFORCE_EAGER, MAX_MODEL_LEN, INTEGRATION_PATH,
                     RESULTS_DIR)

# subprocess.Popen(..., stdout=subprocess.PIPE) with nobody ever reading that
# pipe is a real deadlock risk: vLLM/LMCache log a lot during startup, the
# OS pipe buffer (~64KB) fills, and the child process blocks on its own
# write() call — indistinguishable from "still loading" until you check
# nvidia-smi. Writing to a file instead has no such buffer limit, and
# doubles as a persistent log you can `tail -f`.
def _open_log(name: str):
    path = RESULTS_DIR / f"{name}.log"
    return open(path, "w"), path

logger = logging.getLogger("vllm_launcher")

CACHEBLEND_L1_SIZE_GB = float(os.environ.get("EXP_LMCACHE_L1_SIZE_GB", "8"))


# ── In-process patch (verbatim from the blend_kv_v1 README) ────────────────
_INPROCESS_PATCH_COMMENT_OUT = "ensure_kv_transfer_initialized(vllm_config)"
_INPROCESS_PATCH_INSERT_IMPORTS_AND_CALL = (
    "    from lmcache.v1.compute.models.utils import VLLMModelTracker\n"
    "    from lmcache.integration.vllm.utils import ENGINE_NAME\n"
    "    VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.model)\n"
    "    ensure_kv_transfer_initialized(self.vllm_config)\n"
)


class VLLMLaunchError(RuntimeError):
    pass


class ProcessHandle:
    def __init__(self, name: str, proc: subprocess.Popen):
        self.name = name
        self.proc = proc

    def is_alive(self) -> bool:
        return self.proc.poll() is None

    def terminate(self, timeout_s: float = 15.0):
        if not self.is_alive():
            return
        logger.info(f"  Terminating {self.name} (pid={self.proc.pid}) ...")
        self.proc.terminate()
        try:
            self.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            logger.warning(f"  {self.name} didn't exit in {timeout_s}s — killing")
            self.proc.kill()
            self.proc.wait(timeout=5)


class VLLMLauncher:
    def __init__(self, model_name: str = MODEL_NAME, integration_path: str = INTEGRATION_PATH,
                 vllm_extra_args: Optional[List[str]] = None):
        if integration_path not in ("mp", "inprocess"):
            raise ValueError("integration_path must be 'mp' or 'inprocess'")
        self.model_name = model_name
        self.integration_path = integration_path
        self.vllm_extra_args = vllm_extra_args or []
        self.processes: List[ProcessHandle] = []
        self._patched_file: Optional[Path] = None

    # ── MP mode ──────────────────────────────────────────────────────────
    def _build_lmcache_server_args(self) -> List[str]:
        return [
            "lmcache", "server",
            "--host", LMCACHE_SERVER_HOST,
            "--port", str(LMCACHE_SERVER_PORT),
            "--http-port", str(LMCACHE_HTTP_PORT),
            "--engine-type", "blend",
            "--supported-transfer-mode", "auto",   # blend requires lmcache_driven or auto
            "--l1-size-gb", str(CACHEBLEND_L1_SIZE_GB),
            "--eviction-policy", "LRU",
        ]

    def _build_vllm_mp_args(self) -> List[str]:
        kv_transfer_config = (
            '{"kv_connector":"LMCacheMPConnector",'
            '"kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector",'
            '"kv_role":"kv_both",'
            '"kv_connector_extra_config":{"lmcache.mp.host":"tcp://'
            f'{LMCACHE_SERVER_HOST}","lmcache.mp.port":{LMCACHE_SERVER_PORT}}}}}'
        )
        args = [
            "vllm", "serve", self.model_name,
            "--host", VLLM_HOST, "--port", str(VLLM_PORT),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--block-size", str(VLLM_BLOCK_SIZE),          # must be chunk_size/4 = 256/4
            # NOTE: --attention-backend CUSTOM removed. That flag came from
            # older LMCache/Kubernetes-operator docs and predates vLLM's
            # current attention-backend registry (0.26+), which requires
            # any non-builtin backend to be explicitly registered via
            # register_backend() first — "CUSTOM" as a bare string is no
            # longer a valid value and crashes at model-init time. Current
            # LMCache (0.5.2+) integrates via --kv-transfer-config alone;
            # vLLM picks its normal attention backend (auto) and LMCache's
            # connector handles KV read/write independently of that choice.
            "--no-enable-chunked-prefill",
            "--no-async-scheduling",
            "--pipeline-parallel-size", "1",
            "--disable-hybrid-kv-cache-manager",             # MP connector requirement
            "--kv-transfer-config", kv_transfer_config,
        ]
        if ENFORCE_EAGER:
            args.append("--enforce-eager")
        args += self.vllm_extra_args
        return args

    def start_mp(self):
        env = os.environ.copy()
        env.update(LMCACHE_ENV)

        logger.info("Starting LMCache server (MP mode, engine-type=blend) ...")
        lmcache_args = self._build_lmcache_server_args()
        logger.info(f"  $ {' '.join(lmcache_args)}")
        lmcache_log_f, lmcache_log_path = _open_log("lmcache-server")
        logger.info(f"  Logs: tail -f {lmcache_log_path}")
        lmcache_proc = subprocess.Popen(lmcache_args, env=env,
                                        stdout=lmcache_log_f, stderr=subprocess.STDOUT,
                                        text=True)
        self.processes.append(ProcessHandle("lmcache-server", lmcache_proc))
        self._log_files = getattr(self, "_log_files", []) + [lmcache_log_f]
        self._wait_for_port(LMCACHE_SERVER_HOST, LMCACHE_HTTP_PORT, "LMCache server",
                            timeout_s=60, proc=lmcache_proc, log_path=lmcache_log_path)

        logger.info("Starting vLLM (MP connector -> LMCache server) ...")
        vllm_args = self._build_vllm_mp_args()
        logger.info(f"  $ {' '.join(vllm_args)}")
        vllm_log_f, vllm_log_path = _open_log("vllm-server")
        logger.info(f"  Logs: tail -f {vllm_log_path}  (first launch: this can take "
                    f"several minutes — model download + weight load + CUDA graph capture)")
        vllm_proc = subprocess.Popen(vllm_args, env=env,
                                     stdout=vllm_log_f, stderr=subprocess.STDOUT,
                                     text=True)
        self.processes.append(ProcessHandle("vllm", vllm_proc))
        self._log_files.append(vllm_log_f)
        self._wait_for_port(VLLM_HOST, VLLM_PORT, "vLLM server", timeout_s=900,
                            proc=vllm_proc, log_path=vllm_log_path)
        logger.info("MP mode: both processes up.")
        return self

    # ── In-process mode ──────────────────────────────────────────────────
    def _locate_gpu_worker_py(self) -> Path:
        import importlib.util
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            raise VLLMLaunchError("vllm package not importable — cannot locate gpu_worker.py")
        vllm_root = Path(spec.origin).parent
        candidate = vllm_root / "v1" / "worker" / "gpu_worker.py"
        if not candidate.exists():
            raise VLLMLaunchError(f"Expected {candidate} to exist — vLLM version mismatch? "
                                  f"The in-process patch is pinned to the file layout described "
                                  f"in blend_kv_v1's README; verify manually against your "
                                  f"installed vllm version before proceeding.")
        return candidate

    def apply_inprocess_patch(self, backup: bool = True):
        """Applies the exact 5-line edit from the blend_kv_v1 README to the
        installed vllm package. Idempotent: skips if already applied."""
        path = self._locate_gpu_worker_py()
        src = path.read_text()

        if "VLLMModelTracker.register_model" in src:
            logger.info(f"  {path} already patched — skipping")
            self._patched_file = path
            return

        if backup:
            backup_path = path.with_suffix(path.suffix + ".pre_lmcache_patch.bak")
            shutil.copy2(path, backup_path)
            logger.info(f"  Backed up original to {backup_path}")

        # 1. Comment out ensure_kv_transfer_initialized(vllm_config) inside
        #    init_worker_distributed_environment.
        pattern_comment = re.compile(
            r"^(\s*)ensure_kv_transfer_initialized\(vllm_config\)\s*$", re.MULTILINE)
        matches = list(pattern_comment.finditer(src))
        if len(matches) != 1:
            raise VLLMLaunchError(
                f"Expected exactly 1 occurrence of "
                f"'ensure_kv_transfer_initialized(vllm_config)' to comment out inside "
                f"init_worker_distributed_environment, found {len(matches)}. "
                f"Do not apply this patch blind — inspect {path} manually.")
        m = matches[0]
        indent = m.group(1)
        src = src[:m.start()] + f"{indent}# {_INPROCESS_PATCH_COMMENT_OUT}  # patched by vllm_launcher.py" + src[m.end():]

        # 2. At the end of load_model(), insert the model-registration call.
        #    We locate `def load_model(` and insert before the next top-level
        #    (4-space-indent method) `def ` after it, matching "at the end of
        #    the function, on the base level of the function".
        load_model_match = re.search(r"^(    def load_model\(self.*?:\n)", src, re.MULTILINE)
        if not load_model_match:
            raise VLLMLaunchError(f"Could not find 'def load_model(self...' in {path} — "
                                  f"inspect manually before patching.")
        start = load_model_match.end()
        next_def = re.search(r"^    def \w+\(", src[start:], re.MULTILINE)
        insert_at = start + (next_def.start() if next_def else len(src[start:]))
        src = src[:insert_at] + _INPROCESS_PATCH_INSERT_IMPORTS_AND_CALL + "\n" + src[insert_at:]

        path.write_text(src)
        self._patched_file = path
        logger.info(f"  Patched {path} (in-process CacheBlend mode)")

    def revert_inprocess_patch(self):
        if self._patched_file is None:
            return
        backup_path = self._patched_file.with_suffix(self._patched_file.suffix + ".pre_lmcache_patch.bak")
        if backup_path.exists():
            shutil.copy2(backup_path, self._patched_file)
            logger.info(f"  Reverted {self._patched_file} from backup")
        else:
            logger.warning(f"  No backup found at {backup_path} — could not auto-revert")

    def start_inprocess(self):
        self.apply_inprocess_patch()
        env = os.environ.copy()
        env.update(LMCACHE_ENV)
        kv_transfer_config = (
            '{"kv_connector":"LMCacheConnectorV1",'
            '"kv_connector_module_path":"lmcache.integration.vllm.vllm_v1_adapter",'
            '"kv_role":"kv_both"}'
        )
        args = [
            "vllm", "serve", self.model_name,
            "--host", VLLM_HOST, "--port", str(VLLM_PORT),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--block-size", str(VLLM_BLOCK_SIZE),
            # --attention-backend CUSTOM removed — see _build_vllm_mp_args
            # for why (incompatible with vLLM 0.26's backend registry).
            "--no-enable-chunked-prefill",
            "--no-async-scheduling",
            "--pipeline-parallel-size", "1",
            "--kv-transfer-config", kv_transfer_config,
        ]
        if ENFORCE_EAGER:
            args.append("--enforce-eager")
        args += self.vllm_extra_args
        logger.info("Starting vLLM (in-process CacheBlend, single process) ...")
        logger.info(f"  $ {' '.join(args)}")
        log_f, log_path = _open_log("vllm-inprocess")
        logger.info(f"  Logs: tail -f {log_path}  (first launch: this can take "
                    f"several minutes — model download + weight load + CUDA graph capture)")
        proc = subprocess.Popen(args, env=env, stdout=log_f,
                                stderr=subprocess.STDOUT, text=True)
        self.processes.append(ProcessHandle("vllm-inprocess", proc))
        self._log_files = getattr(self, "_log_files", []) + [log_f]
        self._wait_for_port(VLLM_HOST, VLLM_PORT, "vLLM server", timeout_s=900,
                            proc=proc, log_path=log_path)
        logger.info("In-process mode: vLLM up.")
        return self

    # ── Shared ───────────────────────────────────────────────────────────
    def start(self):
        if self.integration_path == "mp":
            return self.start_mp()
        return self.start_inprocess()

    @staticmethod
    def _wait_for_port(host: str, port: int, label: str, timeout_s: float = 120.0,
                       poll_s: float = 2.0, proc: Optional[subprocess.Popen] = None,
                       log_path: Optional[Path] = None):
        import socket
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if proc is not None and proc.poll() is not None:
                hint = f" — see {log_path} for the actual error" if log_path else ""
                raise VLLMLaunchError(
                    f"{label} process exited early (code {proc.returncode}) before "
                    f"opening {host}:{port}{hint}")
            try:
                with socket.create_connection((host, port), timeout=2):
                    logger.info(f"  {label} is up on {host}:{port}")
                    return
            except OSError:
                time.sleep(poll_s)
        raise VLLMLaunchError(f"{label} did not open {host}:{port} within {timeout_s}s")

    def health_check(self) -> bool:
        try:
            r = requests.get(f"http://{VLLM_HOST}:{VLLM_PORT}/health", timeout=10)
            return r.ok
        except requests.RequestException:
            return False

    def teardown(self):
        for handle in reversed(self.processes):
            handle.terminate()
        for f in getattr(self, "_log_files", []):
            try:
                f.close()
            except Exception:
                pass
        if self.integration_path == "inprocess":
            self.revert_inprocess_patch()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.teardown()
