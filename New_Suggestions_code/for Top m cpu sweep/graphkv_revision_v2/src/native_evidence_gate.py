"""Validate LMCache exact-prefix reuse against an independent native-cache control.

The original numerical gate is intentionally preserved.  If it fails only
because cached and uncached executions differ numerically, this validator
requires the LMCache-warm outputs to match native vLLM prefix-cache outputs
exactly for all saved cases and repeats.  This authorizes only the tested
exact-prefix microbenchmark, never concatenated-context CacheBlend reuse.
"""

import hashlib
import json
import math
from pathlib import Path

from revision_gate import environment_key
from revision_io import atomic, digest


def exact(left, right):
    a, b = left.get("logprobs"), right.get("logprobs")
    return bool(
        left.get("tokens")
        and left.get("tokens") == right.get("tokens")
        and left.get("text") == right.get("text")
        and left.get("prompt_hash") == right.get("prompt_hash")
        and left.get("prompt_tokens") == right.get("prompt_tokens")
        and a
        and b
        and len(a) == len(b) == len(left["tokens"])
        and all(
            x is not None
            and y is not None
            and math.isfinite(x)
            and math.isfinite(y)
            and x == y
            for x, y in zip(a, b)
        )
    )


def validate(gate, control):
    def require(condition, message):
        if not condition:
            raise RuntimeError("Native-control validation failed: " + message)

    require(control.get("status") == "complete_needs_review", "control incomplete")
    require(control.get("gate_sha256") == digest(gate), "control is for another gate")
    require(control.get("environment") == gate.get("environment"), "environments differ")
    conditions = gate.get("conditions", [])
    cache_off = control.get("runs", {}).get("native_cache_off", [])
    cache_on = control.get("runs", {}).get("native_cache_on", [])
    require(len(conditions) == len(cache_off) == len(cache_on) == 3, "three cases required")
    for index, (condition, uncached, cached) in enumerate(zip(conditions, cache_off, cache_on)):
        require(condition["case"] == uncached["case"] == cached["case"] == index, "case order differs")
        require(condition["retrieved_tokens"] > 0, "missing LMCache reuse")
        require(
            condition["shifted_retrieved_tokens"] <= condition["shared_prefix_tokens"],
            "unexpected shifted-prefix reuse",
        )
        require(exact(condition["shifted_cold"], condition["shifted_after_population"]), "shifted outputs differ")
        require(condition["cold"]["tokens"] == condition["warm"]["tokens"], "LMCache changes generated tokens")
        require(condition["cold"]["prompt_hash"] == condition["warm"]["prompt_hash"], "LMCache prompt differs")
        require(exact(condition["cold"], uncached["first"]), "uncached reference differs")
        require(exact(condition["cold"], cached["first"]), "native cold reference differs")
        require(len(uncached["repeats"]) == len(cached["repeats"]) == 3, "three repeats required")
        for repeat in uncached["repeats"]:
            require(exact(condition["cold"], repeat["sample"]), "uncached output differs")
            require(repeat.get("native_prefix_hit_tokens") in (None, 0), "uncached control has hits")
        for repeat in cached["repeats"]:
            require(exact(condition["warm"], repeat["sample"]), "LMCache differs from native cached output")
            require(
                repeat.get("native_prefix_hit_tokens") == condition["retrieved_tokens"],
                "cache-hit lengths differ",
            )
    return {
        "validated": True,
        "criterion": "exact native-cache equivalence of returned token log-probabilities",
        "cases": 3,
        "native_repeats_per_case": 3,
        "scope": "tested exact-prefix requests/model/runtime only; not full-distribution or concatenated-RAG proof",
    }


def require_gate(path, runtime):
    path = Path(path)
    gate = json.loads(path.read_text())
    if gate.get("environment") != environment_key(runtime):
        raise RuntimeError("Gate environment/code changed; rerun controls for the new configuration")
    if gate.get("passed"):
        return digest(gate["environment"])
    control_path = path.parent.parent / "native_cache_control/control_report.json"
    if not control_path.exists():
        raise RuntimeError("Original gate failed; run the independent native-cache control before benchmarking")
    control = json.loads(control_path.read_text())
    verdict = validate(gate, control)
    verdict.update(
        original_gate_passed=gate.get("passed", False),
        gate_hash=digest(gate),
        control_hash=digest(control),
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    atomic(path.parent / "native_equivalence_review.json", verdict)
    print("Validated exact-prefix reuse against native cache controls; original gate preserved.", flush=True)
    return digest({"environment": gate["environment"], "review": verdict})
