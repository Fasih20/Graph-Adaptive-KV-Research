"""HTTP client, prompt parity, request timing, and generic telemetry capture."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import requests


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def parse_prometheus(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        try:
            number = float(parts[1])
        except ValueError:
            continue
        if math.isfinite(number):
            values[parts[0]] = number
    return values


def metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        key: after[key] - before.get(key, 0.0)
        for key in after
        if after[key] - before.get(key, 0.0) != 0.0
    }


def select_metric_sum(metrics: dict[str, float], include: tuple[str, ...]) -> float | None:
    matched = [value for key, value in metrics.items() if all(token in key.lower() for token in include)]
    return float(sum(matched)) if matched else None


@dataclass(frozen=True)
class RequestTiming:
    ttft_ms: float
    request_e2e_ms: float
    prompt_tokens: int
    prompt_hash: str
    output_text_hash: str
    response_id: str | None


class RealCacheClient:
    def __init__(
        self,
        *,
        tokenizer,
        model: str,
        vllm_base_url: str,
        lmcache_http_url: str,
        lmcache_metrics_url: str,
        blend_separator: str,
        gpu_id: str,
        timeout_s: float = 180,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.vllm_base_url = vllm_base_url.rstrip("/")
        self.lmcache_http_url = lmcache_http_url.rstrip("/")
        self.lmcache_metrics_url = lmcache_metrics_url
        self.gpu_id = str(gpu_id)
        self.timeout_s = float(timeout_s)
        self.session = requests.Session()
        # Match LMCache's SegmentTokenDatabase contract exactly.  LMCache
        # currently derives separator IDs with tokenizer.encode(value)[1:].
        # This is observably different from add_special_tokens=False for
        # tokenizers such as Qwen that do not prepend a BOS token.
        self.separator_ids = tokenizer.encode(blend_separator)[1:]
        if not self.separator_ids:
            raise ValueError("blend separator tokenized to an empty sequence")

    def prompt_ids(self, texts: list[str], max_tokens_per_chunk: int = 512) -> list[int]:
        ids: list[int] = []
        bos = self.tokenizer.bos_token_id
        if bos is not None:
            ids.append(int(bos))
        for index, text in enumerate(texts):
            if index:
                ids.extend(self.separator_ids)
            chunk = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_tokens_per_chunk,
            )
            ids.extend(map(int, chunk))
        return ids

    def completion(self, token_ids: list[int], max_tokens: int = 1) -> RequestTiming:
        body = {
            "model": self.model,
            "prompt": token_ids,
            "max_tokens": int(max_tokens),
            # These requests exist to execute and measure prefill, not to
            # evaluate generated text.  With max_tokens=1 a model may select
            # EOS immediately; vLLM then legitimately streams a choice whose
            # decoded text is empty.  Force the requested token count and keep
            # special tokens visible so that this cannot masquerade as a
            # failed request.
            "min_tokens": int(max_tokens),
            "ignore_eos": True,
            "skip_special_tokens": False,
            "temperature": 0,
            "stream": True,
        }
        start = time.perf_counter_ns()
        first_token_ns = None
        first_choice_ns = None
        output_parts: list[str] = []
        response_id = None
        with self.session.post(
            f"{self.vllm_base_url}/v1/completions",
            json=body,
            stream=True,
            timeout=self.timeout_s,
        ) as response:
            response.raise_for_status()
            for raw in response.iter_lines():
                if not raw or not raw.startswith(b"data:"):
                    continue
                payload = raw[5:].strip()
                if payload == b"[DONE]":
                    break
                item = json.loads(payload)
                if item.get("error"):
                    raise RuntimeError(f"vLLM stream error: {item['error']}")
                response_id = response_id or item.get("id")
                choices = item.get("choices", [])
                # An initial empty stream envelope is NOT a generated token.
                if any(c.get("text") or c.get("finish_reason") is not None for c in choices) and first_choice_ns is None:
                    first_choice_ns = time.perf_counter_ns()
                text = "".join(choice.get("text", "") for choice in choices)
                if text and first_token_ns is None:
                    first_token_ns = time.perf_counter_ns()
                output_parts.append(text)
        end = time.perf_counter_ns()
        # A successful vLLM completion can contain a generated special token
        # that has no printable text.  For this max_tokens=1 prefill benchmark,
        # receipt of a choice is the correct completion/TTFT boundary.  An SSE
        # stream containing no choice at all remains a hard failure.
        if first_token_ns is None:
            first_token_ns = first_choice_ns
        if first_token_ns is None:
            raise RuntimeError("stream ended without a completion choice")
        return RequestTiming(
            ttft_ms=(first_token_ns - start) / 1e6,
            request_e2e_ms=(end - start) / 1e6,
            prompt_tokens=len(token_ids),
            prompt_hash=sha256_json(token_ids),
            output_text_hash=hashlib.sha256("".join(output_parts).encode()).hexdigest(),
            response_id=response_id,
        )

    def populate(self, texts: list[str]) -> RequestTiming | None:
        if not texts:
            return None
        return self.completion(self.prompt_ids(texts), max_tokens=1)

    def metrics(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {"vllm": {}, "lmcache": {}}
        for name, url in (
            ("vllm", f"{self.vllm_base_url}/metrics"),
            ("lmcache", self.lmcache_metrics_url),
        ):
            try:
                response = self.session.get(url, timeout=10)
                if response.ok:
                    result[name] = parse_prometheus(response.text)
            except requests.RequestException:
                pass
        return result

    def status(self) -> dict:
        try:
            response = self.session.get(f"{self.lmcache_http_url}/status", timeout=10)
            if response.ok:
                return response.json()
        except (requests.RequestException, ValueError):
            pass
        return {}

    def gpu_snapshot(self) -> dict:
        command = [
            "nvidia-smi", f"--id={self.gpu_id}",
            "--query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
        return {"returncode": result.returncode, "row": result.stdout.strip()}

    def close(self) -> None:
        self.session.close()
