#!/usr/bin/env python3
"""Portable entry point for the isolated real-cache benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from runtime_env import PreflightError, run_preflight, stable_config_hash

REPAIRED_K_VALUES = (3, 4, 5, 6, 8, 10, 12, 14, 16)
REPAIRED_POLICIES = (
    "no_prefetch",
    "cosine",
    "graph_fixed",
    "adaptive_offline_global",
    "adaptive_online_warm",
)


def parse_csv(value: str, cast=str):
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def estimate_main_workload(policies, k_values, events: int, repetitions: int) -> dict:
    """Return the real-inference request upper bound before a long run starts."""
    arms = len(policies) * len(k_values) * repetitions
    access_requests = 2 * events * arms  # current access plus target demand
    prefetch_policy_count = sum(policy != "no_prefetch" for policy in policies)
    population_upper_bound = (
        events * repetitions * prefetch_policy_count * sum(k_values)
    )
    return {
        "policies": list(policies),
        "k_values": list(k_values),
        "events_per_arm": events,
        "repetitions": repetitions,
        "main_isolated_process_pairs": arms,
        "current_plus_target_requests": access_requests,
        "candidate_population_request_upper_bound": population_upper_bound,
        "total_inference_request_upper_bound": access_requests + population_upper_bound,
        "note": (
            "Upper bound for main arms only. Each candidate is populated by a real "
            "one-token vLLM request; graph policies may return fewer than K candidates. "
            "The four mandatory sanity process pairs are additional."
        ),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="hotpot", choices=["hotpot", "2wiki", "musique", "multifield"])
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--model-revision")
    p.add_argument("--gpu", default="0", help="Physical GPU index or UUID; only this GPU is exposed to children")
    p.add_argument("--policies", default=",".join(REPAIRED_POLICIES))
    p.add_argument("--k-values", default=",".join(map(str, REPAIRED_K_VALUES)))
    p.add_argument("--events", type=int, default=100)
    p.add_argument("--train-events", type=int, default=300)
    p.add_argument("--dev-events", type=int, default=150)
    p.add_argument("--repetitions", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-documents", type=int, default=60)
    p.add_argument("--max-chars-per-document", type=int, default=12000)
    p.add_argument(
        "--document-chunk-tokens",
        type=int,
        default=384,
        help="Maximum tokenizer tokens per graph/document chunk (not LMCache block size)",
    )
    p.add_argument(
        "--document-chunk-overlap-tokens",
        type=int,
        default=64,
        help="Tokenizer-token overlap between consecutive graph/document chunks",
    )
    p.add_argument("--top-m", type=int, default=20)
    p.add_argument("--embedding-device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--overlap-ms", type=float, default=0.0)
    p.add_argument(
        "--l1-size-gb",
        type=float,
        default=0.5,
        help="LMCache L1 capacity; 0.5 GiB intentionally creates pressure in the validation workload",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--vllm-port", type=int, default=8000)
    p.add_argument("--lmcache-port", type=int, default=5555)
    p.add_argument(
        "--lmcache-http-port",
        type=int,
        default=18080,
        help="LMCache HTTP port; 18080 avoids Colab's commonly occupied 8080",
    )
    p.add_argument("--lmcache-prometheus-port", type=int, default=9090)
    p.add_argument("--startup-timeout", type=float, default=900)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--sanity-only", action="store_true")
    p.add_argument("--skip-sanity", action="store_true", help="Expert override: do not run the mandatory fresh-process controls")
    p.add_argument("--rerun-sanity", action="store_true", help="Ignore an existing passing sanity report and rerun all controls")
    p.add_argument("--override-sanity-failure", action="store_true", help="Continue after failed controls; output is labeled non-publication-ready")
    return p


def main() -> int:
    args = parser().parse_args()
    policies = parse_csv(args.policies)
    k_values = parse_csv(args.k_values, int)
    invalid = set(policies) - set(REPAIRED_POLICIES)
    if invalid:
        raise SystemExit(f"unsupported policies: {sorted(invalid)}")
    if not policies or not k_values:
        raise SystemExit("at least one policy and one K value are required")
    if len(set(policies)) != len(policies) or len(set(k_values)) != len(k_values):
        raise SystemExit("duplicate policies or K values are not allowed")
    if args.events < 1 or args.repetitions < 1 or any(k < 1 for k in k_values):
        raise SystemExit("events, repetitions, and every K must be positive")
    if args.embedding_device != "cpu":
        raise SystemExit(
            "The isolated benchmark requires --embedding-device cpu so the parent process "
            "cannot retain CUDA allocations that compete with vLLM."
        )
    if args.chunk_size % args.block_size:
        raise SystemExit("--chunk-size must be a multiple of --block-size")
    if args.document_chunk_tokens < 1:
        raise SystemExit("--document-chunk-tokens must be positive")
    if not 0 <= args.document_chunk_overlap_tokens < args.document_chunk_tokens:
        raise SystemExit(
            "--document-chunk-overlap-tokens must be non-negative and smaller than "
            "--document-chunk-tokens"
        )
    if args.document_chunk_tokens + 1 > args.max_model_len:
        raise SystemExit(
            "--document-chunk-tokens must leave room for at least one generated token "
            "inside --max-model-len"
        )
    if args.l1_size_gb <= 0:
        raise SystemExit("--l1-size-gb must be positive")

    safe_model = args.model.replace("/", "_").replace("-", "_")
    output = args.output_dir or ROOT / "outputs" / f"isolated_{args.dataset}_{safe_model}"
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(output / "driver.log")],
    )
    workload_scale = estimate_main_workload(
        policies, k_values, args.events, args.repetitions
    )
    logging.info(
        "Planned main workload: %d isolated arms; up to %s real vLLM requests "
        "(%s candidate-population + %s current/target).",
        workload_scale["main_isolated_process_pairs"],
        f'{workload_scale["total_inference_request_upper_bound"]:,}',
        f'{workload_scale["candidate_population_request_upper_bound"]:,}',
        f'{workload_scale["current_plus_target_requests"]:,}',
    )
    if workload_scale["total_inference_request_upper_bound"] > 100_000:
        logging.warning(
            "This is a very large real-inference workload. Run a reduced-K/model "
            "screen first and preserve this estimate with the results."
        )
    try:
        report = run_preflight(
            project_root=ROOT,
            output_dir=output,
            gpu_id=args.gpu,
            vllm_host="127.0.0.1",
            vllm_port=args.vllm_port,
            lmcache_host="127.0.0.1",
            lmcache_port=args.lmcache_port,
            lmcache_http_port=args.lmcache_http_port,
            lmcache_prometheus_port=args.lmcache_prometheus_port,
        )
    except PreflightError as exc:
        logging.error("Preflight failed: %s", exc)
        return 2
    logging.info("Preflight passed on %s; child CUDA_VISIBLE_DEVICES=%s", report.selected_gpu["name"], args.gpu)
    if args.preflight_only:
        return 0

    scientific_config = {
        name: value for name, value in vars(args).items()
        if name not in {
            "output_dir", "preflight_only", "sanity_only", "skip_sanity",
            "rerun_sanity", "override_sanity_failure",
        }
    }
    scientific_config.update({"policies": policies, "k_values": k_values})
    config_hash = stable_config_hash(scientific_config)
    existing_manifest = output / "run_manifest.json"
    if existing_manifest.exists():
        previous = json.loads(existing_manifest.read_text())
        if previous.get("config_hash") != config_hash:
            raise SystemExit(
                "Output directory already belongs to a different benchmark configuration. "
                "Choose a new --output-dir; existing arm results were not changed."
            )
    (output / "run_scale_estimate.json").write_text(
        json.dumps(workload_scale, indent=2), encoding="utf-8"
    )

    # CPU preprocessing happens before vLLM starts so MiniLM never competes
    # with the causal model for T4 memory.  CUDA is available only if the user
    # explicitly opts into --embedding-device cuda.
    from transformers import AutoConfig, AutoTokenizer
    from chunk_store import ChunkStore
    from isolated_runtime import RuntimeConfig
    from repaired_benchmark import BenchmarkConfig, run_benchmark
    from sanity_controls import run_sanity_controls

    runtime = RuntimeConfig(
        model=args.model,
        model_revision=args.model_revision,
        gpu_id=args.gpu,
        vllm_host="127.0.0.1",
        vllm_port=args.vllm_port,
        lmcache_host="127.0.0.1",
        lmcache_port=args.lmcache_port,
        lmcache_http_port=args.lmcache_http_port,
        lmcache_prometheus_port=args.lmcache_prometheus_port,
        max_model_len=args.max_model_len,
        block_size=args.block_size,
        chunk_size=args.chunk_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        l1_size_gb=args.l1_size_gb,
        startup_timeout_s=args.startup_timeout,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
    )

    model_config = AutoConfig.from_pretrained(
        args.model, revision=args.model_revision, trust_remote_code=args.trust_remote_code
    )
    resolved_revision = getattr(model_config, "_commit_hash", None) or args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=resolved_revision, trust_remote_code=args.trust_remote_code
    )
    runtime = RuntimeConfig(**{**asdict(runtime), "model_revision": resolved_revision})
    store = ChunkStore(
        args.dataset,
        tokenizer=tokenizer,
        model_config=model_config,
        top_m=args.top_m,
        max_documents=args.max_documents,
        max_chars_per_document=args.max_chars_per_document,
        document_chunk_tokens=args.document_chunk_tokens,
        document_chunk_overlap_tokens=args.document_chunk_overlap_tokens,
        embedding_device=args.embedding_device,
        seed=args.seed,
    ).prepare()
    kv_sizes = [chunk.kv_bytes for chunk in store.chunks]
    token_sizes = [chunk.token_count for chunk in store.chunks]
    l1_capacity_bytes = int(runtime.l1_size_gb * 1024 ** 3)
    corpus_kv_bytes = sum(kv_sizes)
    (output / "workload_stats.json").write_text(
        json.dumps(
            {
                "chunks": len(store.chunks),
                "documents": len({chunk.document_id for chunk in store.chunks}),
                "mean_chunk_kv_bytes": sum(kv_sizes) / len(kv_sizes),
                "median_chunk_kv_bytes": sorted(kv_sizes)[len(kv_sizes) // 2],
                "min_chunk_tokens": min(token_sizes),
                "mean_chunk_tokens": sum(token_sizes) / len(token_sizes),
                "median_chunk_tokens": sorted(token_sizes)[len(token_sizes) // 2],
                "max_chunk_tokens": max(token_sizes),
                "document_chunk_tokens_configured": args.document_chunk_tokens,
                "document_chunk_overlap_tokens_configured": args.document_chunk_overlap_tokens,
                "l1_capacity_bytes": l1_capacity_bytes,
                "corpus_kv_bytes": corpus_kv_bytes,
                "corpus_to_l1_ratio": corpus_kv_bytes / l1_capacity_bytes,
                "approximate_mean_chunk_capacity": (
                    l1_capacity_bytes / (sum(kv_sizes) / len(kv_sizes))
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    traces = {
        "train": store.generate_events("train", args.train_events, 0),
        "dev": store.generate_events("dev", args.dev_events, 1),
        "test": store.generate_events("test", args.events, 2),
    }
    store.save(output / "workload", traces)
    sanity_path = output / "sanity" / "sanity_report.json"
    if not args.skip_sanity:
        if sanity_path.exists() and not args.rerun_sanity:
            sanity = json.loads(sanity_path.read_text())
            if (
                sanity.get("runtime") != asdict(runtime)
                or sanity.get("workload_fingerprint") != store.workload_fingerprint()
                or int(sanity.get("configured_repetitions", -1)) != args.repetitions
            ):
                raise SystemExit(
                    "Existing sanity report was produced with a different runtime or workload. "
                    "Use --rerun-sanity or a new --output-dir."
                )
            if sanity.get("critical_passed") or args.override_sanity_failure:
                logging.info("Reusing existing sanity report: %s", sanity_path)
            else:
                logging.warning(
                    "Existing sanity report failed; rerunning controls with the "
                    "current checker instead of reusing the failure."
                )
                sanity = run_sanity_controls(
                    output / "sanity",
                    runtime,
                    store,
                    traces["test"][0],
                    repetitions=args.repetitions,
                )
        else:
            sanity = run_sanity_controls(
                output / "sanity",
                runtime,
                store,
                traces["test"][0],
                repetitions=args.repetitions,
            )
        if not sanity["critical_passed"] and not args.override_sanity_failure:
            logging.error("Sanity controls failed. Inspect %s", output / "sanity" / "sanity_report.json")
            return 2
        if args.sanity_only:
            return 0 if sanity["critical_passed"] else 2
    elif args.sanity_only:
        raise SystemExit("--sanity-only cannot be combined with --skip-sanity")
    benchmark = BenchmarkConfig(
        output_dir=output,
        policies=policies,
        k_values=k_values,
        repetitions=args.repetitions,
        seed=args.seed,
        overlap_ms=args.overlap_ms,
        max_events=args.events,
        max_tokens_per_chunk=args.document_chunk_tokens,
        runtime=runtime,
    )
    manifest = {
        "config": {**scientific_config, "output_dir": str(output), "skip_sanity": args.skip_sanity},
        "runtime": asdict(runtime),
        "config_hash": config_hash,
        "measurement_semantics": "exact_chunk_access_prefetch_microbenchmark",
        "cache_semantics": "stock_lmcache_exact_prefix_per_chunk_not_cacheblend",
        "trace_semantics": "controlled_synthetic_document_split",
        "warning": "Speculative population uses real inference and the access targets are controlled synthetic diagnostics. Population cost is included end-to-end; neither is claimed as production trace-driven transfer prefetching.",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    summary = run_benchmark(benchmark, store, traces)
    logging.info("Complete: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
