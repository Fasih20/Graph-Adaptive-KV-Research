"""
llm_providers.py
=================
Thin, dependency-free (just `requests`) unified caller across three provider
families, selected by a --provider tag:

  gemini      -> Google AI Studio Gemini API (generateContent)
  openai      -> anything speaking the OpenAI chat-completions wire format —
                 OpenRouter, Groq, or OpenAI itself, distinguished by
                 --backend (openrouter | groq | openai) which just changes
                 base_url + api-key env var
  anthropic   -> Anthropic Messages API

Free-tier reality check (verified at delivery time — re-check before a long
run, these move fast):
  - gemini:    genuinely free (Flash / Flash-Lite only — Pro is paid-only
               as of Apr 2026), rate-limited not token-limited, no card.
  - openrouter: genuinely free for :free-suffixed models, 20 RPM / 50 RPD
               (or 1000/day if you've ever topped up $10 — optional).
  - groq:      genuinely free, ~30 RPM / 1000 RPD per model, no card.
  - anthropic: NO ongoing free tier. New console.anthropic.com accounts get
               a one-time ~$5 trial credit (phone verification, no card) —
               fine for a small run on Haiku, not a permanent free lane.

Model names below are current-as-of-writing defaults, not guarantees —
free-tier model rosters (especially OpenRouter's `:free` list) rotate.
Override with --model if a default has gone stale.
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger("llm_providers")

GEMINI_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

BACKEND_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
}
BACKEND_API_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# cheap = safe on a free tier (small/fast model); full = higher-quality
# default for a heavier, presumably-paid or larger-budget run.
MODEL_DEFAULTS = {
    "gemini":    {"cheap": "gemini-2.5-flash-lite", "full": "gemini-2.5-flash"},
    "openrouter": {"cheap": "meta-llama/llama-3.3-70b-instruct:free", "full": "meta-llama/llama-3.3-70b-instruct:free"},
    "groq":      {"cheap": "llama-3.3-70b-versatile", "full": "llama-3.3-70b-versatile"},
    "openai":    {"cheap": "gpt-4o-mini", "full": "gpt-4o"},
    "anthropic": {"cheap": "claude-haiku-4-5", "full": "claude-sonnet-4-6"},
}


@dataclass
class LLMResponse:
    text: str
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


def resolve_model(provider: str, backend: Optional[str], cheap: bool, override: Optional[str]) -> str:
    if override:
        return override
    key = backend if provider == "openai" else provider
    return MODEL_DEFAULTS.get(key, {}).get("cheap" if cheap else "full") or MODEL_DEFAULTS["openai"]["cheap"]


def _post_with_retry(url, headers=None, json=None, params=None, max_retries=5, timeout=60):
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        resp = requests.post(url, headers=headers, json=json, params=params, timeout=timeout)
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == max_retries:
                resp.raise_for_status()
            logger.warning(f"  {resp.status_code} from {url.split('?')[0]} — retry {attempt}/{max_retries} "
                          f"in {delay:.0f}s (free-tier rate limit is the usual cause)")
            time.sleep(delay)
            delay = min(delay * 2, 60)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError("unreachable")


def call_gemini(model: str, prompt: str, api_key: str, max_tokens: int, temperature: float = 0.0) -> LLMResponse:
    url = GEMINI_URL_TMPL.format(model=model)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
    }
    resp = _post_with_retry(url, params={"key": api_key}, json=body)
    data = resp.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        # Common cause: finishReason=MAX_TOKENS with no text yet, or a
        # safety block — surface the raw response for debugging.
        text = ""
        logger.warning(f"  Gemini returned no text — raw: {data}")
    usage = data.get("usageMetadata", {})
    return LLMResponse(text=text, input_tokens=usage.get("promptTokenCount"),
                       output_tokens=usage.get("candidatesTokenCount"))


def call_openai_format(base_url: str, model: str, prompt: str, api_key: str,
                       max_tokens: int, temperature: float = 0.0) -> LLMResponse:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature}
    resp = _post_with_retry(url, headers=headers, json=body)
    data = resp.json()
    text = data["choices"][0]["message"]["content"] or ""
    usage = data.get("usage", {})
    return LLMResponse(text=text, input_tokens=usage.get("prompt_tokens"),
                       output_tokens=usage.get("completion_tokens"))


def call_anthropic(model: str, prompt: str, api_key: str, max_tokens: int,
                   temperature: float = 0.0) -> LLMResponse:
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    body = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}]}
    resp = _post_with_retry(url, headers=headers, json=body)
    data = resp.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    usage = data.get("usage", {})
    return LLMResponse(text=text, input_tokens=usage.get("input_tokens"),
                       output_tokens=usage.get("output_tokens"))


class LLMClient:
    """Provider-agnostic entrypoint. provider in {'gemini','openai','anthropic'};
    backend only matters (and is required) when provider == 'openai'."""

    def __init__(self, provider: str, backend: Optional[str] = None,
                model: Optional[str] = None, cheap: bool = True,
                api_key: Optional[str] = None):
        self.provider = provider
        self.backend = backend
        if provider == "openai":
            if backend not in BACKEND_BASE_URLS:
                raise ValueError(f"--backend must be one of {list(BACKEND_BASE_URLS)} when --provider openai")
            self.base_url = BACKEND_BASE_URLS[backend]
            env_var = BACKEND_API_KEY_ENV[backend]
        elif provider == "gemini":
            env_var = "GEMINI_API_KEY"
        elif provider == "anthropic":
            env_var = "ANTHROPIC_API_KEY"
        else:
            raise ValueError(f"Unknown provider {provider!r}")

        self.api_key = api_key or os.environ.get(env_var)
        if not self.api_key:
            raise ValueError(f"No API key — pass --api_key or set ${env_var}")

        self.model = resolve_model(provider, backend, cheap, model)
        logger.info(f"  LLM: provider={provider}"
                    f"{f' backend={backend}' if backend else ''} model={self.model}")

    def generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.0) -> LLMResponse:
        if self.provider == "gemini":
            return call_gemini(self.model, prompt, self.api_key, max_tokens, temperature)
        if self.provider == "openai":
            return call_openai_format(self.base_url, self.model, prompt, self.api_key, max_tokens, temperature)
        if self.provider == "anthropic":
            return call_anthropic(self.model, prompt, self.api_key, max_tokens, temperature)
        raise ValueError(self.provider)
