"""Version-light Gemini REST client with bounded, server-aware retries."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

import requests


LOGGER = logging.getLogger("graphkv_quality.provider")
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"


class ProviderError(RuntimeError):
    pass


class QuotaExhaustedError(ProviderError):
    pass


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    attempts: int
    finish_reason: str | None


class RollingRateLimiter:
    """Optional rolling RPM cap. A value of zero disables proactive sleeps."""

    def __init__(self, rpm: int = 0):
        self.rpm = max(0, int(rpm))
        self.timestamps: list[float] = []

    def wait(self) -> None:
        if self.rpm == 0:
            return
        now = time.monotonic()
        self.timestamps = [stamp for stamp in self.timestamps if stamp > now - 60.0]
        if len(self.timestamps) >= self.rpm:
            delay = self.timestamps[0] + 60.0 - now
            if delay > 0:
                LOGGER.info("RPM cap reached; waiting %.1fs", delay)
                time.sleep(delay)
            now = time.monotonic()
            self.timestamps = [stamp for stamp in self.timestamps if stamp > now - 60.0]
        self.timestamps.append(time.monotonic())


def _retry_after_seconds(response: requests.Response, fallback: float) -> float:
    header = response.headers.get("Retry-After")
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            pass
    try:
        details = response.json().get("error", {}).get("details", [])
        for detail in details:
            retry_delay = detail.get("retryDelay")
            if retry_delay:
                match = re.fullmatch(r"([0-9.]+)s", str(retry_delay))
                if match:
                    return max(0.0, float(match.group(1)))
    except (ValueError, AttributeError):
        pass
    return fallback


def _safe_error(response: requests.Response) -> str:
    try:
        payload: Any = response.json()
        message = payload.get("error", {}).get("message", payload)
        return str(message)[:1000]
    except (ValueError, AttributeError):
        return response.text[:1000]


class GeminiClient:
    """Direct Gemini generateContent caller; no Google SDK version coupling."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        rpm: int = 0,
        max_retries: int = 4,
        connect_timeout: float = 10.0,
        read_timeout: float = 90.0,
        max_retry_wait: float = 90.0,
        session: requests.Session | None = None,
    ):
        self.model = str(model)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Set GEMINI_API_KEY (preferred), GOOGLE_API_KEY, or pass --api-key")
        self.rate_limiter = RollingRateLimiter(rpm)
        self.max_retries = max(1, int(max_retries))
        self.timeout = (float(connect_timeout), float(read_timeout))
        self.max_retry_wait = max(1.0, float(max_retry_wait))
        self.session = session or requests.Session()

    @property
    def model_url(self) -> str:
        return f"{API_ROOT}/models/{self.model}"

    def preflight(self) -> dict:
        response = self.session.get(self.model_url, params={"key": self.api_key}, timeout=self.timeout)
        if not response.ok:
            raise ProviderError(
                f"Gemini model preflight failed ({response.status_code}): {_safe_error(response)}"
            )
        model = response.json()
        methods = set(model.get("supportedGenerationMethods", []))
        if "generateContent" not in methods:
            raise ProviderError(f"Model {self.model!r} does not advertise generateContent support")
        return {
            "name": model.get("name", self.model),
            "display_name": model.get("displayName"),
            "input_token_limit": model.get("inputTokenLimit"),
            "output_token_limit": model.get("outputTokenLimit"),
            "supported_generation_methods": sorted(methods),
        }

    def generate(self, prompt: str, *, max_output_tokens: int = 64) -> Generation:
        url = f"{self.model_url}:generateContent"
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "candidateCount": 1,
                "temperature": 0.0,
                "maxOutputTokens": int(max_output_tokens),
            },
        }
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self.rate_limiter.wait()
            try:
                response = self.session.post(
                    url,
                    params={"key": self.api_key},
                    json=body,
                    timeout=self.timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as error:
                last_error = error
                if attempt == self.max_retries:
                    break
                delay = min(self.max_retry_wait, (2 ** (attempt - 1)) + random.random())
                LOGGER.warning("Transient Gemini connection error; retrying in %.1fs: %s", delay, error)
                time.sleep(delay)
                continue

            if response.status_code == 429 or response.status_code >= 500:
                last_error = ProviderError(f"HTTP {response.status_code}: {_safe_error(response)}")
                if attempt == self.max_retries:
                    if response.status_code == 429:
                        raise QuotaExhaustedError(
                            f"Gemini returned HTTP 429 after {attempt} attempts. Progress is checkpointed. "
                            "Check this project's active RPM/RPD in AI Studio, wait for reset, then rerun "
                            "the identical command."
                        )
                    break
                fallback = min(self.max_retry_wait, 2 ** (attempt - 1))
                delay = min(self.max_retry_wait, _retry_after_seconds(response, fallback))
                LOGGER.warning("Gemini HTTP %d; retrying in %.1fs", response.status_code, delay)
                time.sleep(delay)
                continue

            if not response.ok:
                raise ProviderError(f"Gemini HTTP {response.status_code}: {_safe_error(response)}")

            payload = response.json()
            candidates = payload.get("candidates") or []
            if not candidates:
                raise ProviderError(f"Gemini returned no candidates: {json.dumps(payload)[:1200]}")
            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts", [])
            text = "".join(str(part.get("text", "")) for part in parts).strip()
            finish_reason = candidate.get("finishReason")
            if not text:
                raise ProviderError(
                    f"Gemini returned an empty answer (finishReason={finish_reason!r}): "
                    f"{json.dumps(payload)[:1200]}"
                )
            usage = payload.get("usageMetadata", {})
            return Generation(
                text=text,
                input_tokens=usage.get("promptTokenCount"),
                output_tokens=usage.get("candidatesTokenCount"),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                attempts=attempt,
                finish_reason=finish_reason,
            )

        raise ProviderError(f"Gemini request failed after {self.max_retries} attempts: {last_error}")


def list_gemini_models(api_key: str | None = None) -> list[dict]:
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    response = requests.get(f"{API_ROOT}/models", params={"key": key, "pageSize": 1000}, timeout=(10, 60))
    if not response.ok:
        raise ProviderError(f"Could not list Gemini models ({response.status_code}): {_safe_error(response)}")
    rows = []
    for model in response.json().get("models", []):
        methods = model.get("supportedGenerationMethods", [])
        if "generateContent" in methods:
            rows.append(
                {
                    "name": str(model.get("name", "")).removeprefix("models/"),
                    "display_name": model.get("displayName"),
                    "input_token_limit": model.get("inputTokenLimit"),
                    "output_token_limit": model.get("outputTokenLimit"),
                }
            )
    return sorted(rows, key=lambda row: row["name"])
