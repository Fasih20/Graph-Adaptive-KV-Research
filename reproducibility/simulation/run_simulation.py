#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from graphkv_sim.analysis import load_result_csvs, write_analysis
from graphkv_sim.cost import CostModel
from graphkv_sim.fit import fit_global_weights, fit_query_type_oracle_weights, tune_online_eta
from graphkv_sim.legacy_graph import build_graph_document_aware
from graphkv_sim.longbench import DATASETS, load_longbench_documents
from graphkv_sim.policies import (
    CosinePolicy,
    FixedGraphPolicy,
    LinearGraphPolicy,
    NoPrefetchPolicy,
    OnlineAdaptivePolicy,
    OraclePolicy,
    QueryTypeOraclePolicy,
    RandomPolicy,
    RelativeOffsetTransitionPolicy,
    SequentialPolicy,
)
from graphkv_sim.runner import run_policy_trace
from graphkv_sim.traces import (
    ModelShape,
    balanced_event_sample,
    build_chunks,
    cosine_similarity_matrix,
    document_orders,
    grouped_synthetic_events,
    load_trace_csv,
    save_chunks_jsonl,
    save_trace_csv,
    split_by_document,
    split_document_ids,
    stratified_ablation_events,
)


FULL_K_VALUES = [3, 4, 5, 6, 8, 10, 12, 14, 16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe GraphKV cache simulator")
    parser.add_argument("--dataset", choices=DATASETS, default="hotpot")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-documents", type=int, default=12)
    parser.add_argument("--max-chars-per-document", type=int, default=18000)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--trace-csv", help="Independent trace; omit for labelled grouped-synthetic trace")
    parser.add_argument("--max-test-events", type=int, default=0)
    parser.add_argument("--k", nargs="+", type=int, default=[3, 8, 16])
    parser.add_argument("--top-m", type=int, default=20)
    parser.add_argument("--capacity-entries", type=int, default=32)
    parser.add_argument("--capacity-gib", type=float)
    parser.add_argument("--prefetch-mode", choices=["blocking", "deadline"], default="blocking")
    parser.add_argument("--overlap-window-ms", type=float, default=0.0)
    parser.add_argument("--cost-model-json", help="Optional output from calibrate_small_model.py")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/smoke")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Balanced <=20-event, K=3 mechanics check")
    mode.add_argument(
        "--stress-smoke",
        action="store_true",
        help="Balanced smoke with tiny cache/deadline to exercise evictions and late prefetch",
    )
    mode.add_argument(
        "--full-comparison-600",
        action="store_true",
        help="Held-out 600-query, nine-K compatibility ablation",
    )
    parser.add_argument("--comparison-train-events", type=int, default=600)
    parser.add_argument("--comparison-dev-events", type=int, default=200)
    parser.add_argument("--comparison-test-events", type=int, default=600)
    parser.add_argument(
        "--comparison-max-chars-per-document",
        type=int,
        default=4000,
        help="30 documents x 4000 chars approximates the old ~100k-character corpus scale",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke or args.stress_smoke:
        args.max_documents = max(6, min(args.max_documents, 8))
        args.max_chars_per_document = min(args.max_chars_per_document, 8000)
        args.max_test_events = 20
        args.k = [3]
    if args.stress_smoke:
        args.capacity_entries = 2
        args.prefetch_mode = "deadline"
        args.overlap_window_ms = 0.02
    if args.full_comparison_600:
        if args.trace_csv:
            raise ValueError("--full-comparison-600 generates its own compatibility trace")
        args.max_documents = max(args.max_documents, 30)
        args.max_chars_per_document = min(
            args.max_chars_per_document, args.comparison_max_chars_per_document
        )
        args.k = FULL_K_VALUES
        args.capacity_entries = 16
        args.max_test_events = 0
        args.checkpoint_every = max(args.checkpoint_every, 25)
        if args.output_dir == "outputs/smoke":
            args.output_dir = "outputs/full_comparison_600"
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("graphkv_sim")

    from transformers import AutoConfig, AutoTokenizer
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer

    logger.info("Loading tokenizer/config for model-aware token and KV-byte accounting")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    shape = ModelShape.from_hf_config(config)
    documents = load_longbench_documents(
        args.dataset, args.max_documents, args.seed, args.max_chars_per_document
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        length_function=len,
    )
    chunks = build_chunks(documents, tokenizer, splitter, shape)
    if len(chunks) < 6:
        raise RuntimeError("too few chunks; increase documents or characters")
    save_chunks_jsonl(chunks, output / "chunks.jsonl")

    logger.info("Embedding %d chunks", len(chunks))
    embedder = SentenceTransformer(args.embedding_model)
    embeddings = embedder.encode(
        [chunk.text for chunk in chunks], batch_size=32, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    similarity = cosine_similarity_matrix(embeddings)
    adjacency, raw_edges, graph_seconds = build_graph_document_aware(
        embeddings,
        similarity,
        [chunk.document_id for chunk in chunks],
        [chunk.position for chunk in chunks],
        logger,
        "topm",
        args.top_m,
    )

    if args.full_comparison_600:
        document_splits = split_document_ids(
            [chunk.document_id for chunk in chunks], args.seed
        )
        proportions = {"semantic": 0.25, "structural": 0.25, "multi-hop": 0.50}
        splits = {
            "train": stratified_ablation_events(
                chunks,
                similarity,
                args.comparison_train_events,
                document_splits["train"],
                seed=args.seed,
                proportions=proportions,
                event_id_start=100000,
                session_prefix="comparison_train",
            ),
            "dev": stratified_ablation_events(
                chunks,
                similarity,
                args.comparison_dev_events,
                document_splits["dev"],
                seed=args.seed + 1,
                proportions=proportions,
                event_id_start=200000,
                session_prefix="comparison_dev",
            ),
            "test": stratified_ablation_events(
                chunks,
                similarity,
                args.comparison_test_events,
                document_splits["test"],
                seed=args.seed + 2,
                proportions=proportions,
                event_id_start=0,
                session_prefix="comparison_test",
            ),
        }
        events = splits["train"] + splits["dev"] + splits["test"]
        trace_kind = "heldout_document_stratified_synthetic_600_compatibility"
    elif args.trace_csv:
        events = load_trace_csv(args.trace_csv)
        trace_kind = "independent_csv"
        splits = split_by_document(events, args.seed)
    else:
        events = grouped_synthetic_events(chunks, similarity, args.seed)
        trace_kind = "grouped_synthetic_diagnostic"
        splits = split_by_document(events, args.seed)
    save_trace_csv(events, output / "trace_all.csv")
    for split_name, split_events in splits.items():
        save_trace_csv(split_events, output / f"trace_{split_name}.csv")
    test_events = splits["test"]
    if args.max_test_events:
        test_events = balanced_event_sample(
            test_events, args.max_test_events, args.seed
        )

    offline_weights, grid_rows = fit_global_weights(splits["train"], raw_edges, args.k)
    query_weights = fit_query_type_oracle_weights(splits["train"], raw_edges, args.k)
    eta, eta_rows = tune_online_eta(splits["dev"], raw_edges, offline_weights, k=args.k[len(args.k) // 2])
    pd.DataFrame(grid_rows).to_csv(output / "offline_weight_grid.csv", index=False)
    pd.DataFrame(eta_rows).to_csv(output / "online_eta_grid.csv", index=False)

    metadata = {
        "model": args.model,
        "embedding_model": args.embedding_model,
        "dataset": args.dataset,
        "trace_kind": trace_kind,
        "run_mode": (
            "full_comparison_600"
            if args.full_comparison_600
            else "stress_smoke"
            if args.stress_smoke
            else "smoke"
            if args.smoke
            else "standard"
        ),
        "seed": args.seed,
        "top_m": args.top_m,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "corpus_documents": len(documents),
        "corpus_characters": int(sum(len(text) for text in documents.values())),
        "corpus_chunks": len(chunks),
        "graph_build_seconds": graph_seconds,
        "model_shape": shape.__dict__,
        "offline_weights": offline_weights,
        "query_type_oracle_weights": query_weights,
        "online_eta": eta,
        "k_values": args.k,
        "capacity_entries": args.capacity_entries,
        "evaluated_test_events": len(test_events),
        "evaluated_test_documents": sorted(
            {event.document_id for event in test_events}
        ),
        "evaluated_query_counts": {
            query_type: sum(event.query_type == query_type for event in test_events)
            for query_type in sorted({event.query_type for event in test_events})
        },
        "comparison_query_proportions": (
            {"semantic": 0.25, "structural": 0.25, "multi-hop": 0.50}
            if args.full_comparison_600
            else None
        ),
        "comparison_is_independent_workload": False if args.full_comparison_600 else None,
        "split_documents": {
            name: sorted({event.document_id for event in values}) for name, values in splits.items()
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    bytes_capacity = int(args.capacity_gib * 1024**3) if args.capacity_gib else None
    if args.cost_model_json:
        calibrated = json.loads(Path(args.cost_model_json).read_text(encoding="utf-8"))["cost_model"]
        calibrated.update(
            prefetch_mode=args.prefetch_mode,
            overlap_window_ms=args.overlap_window_ms,
        )
        cost = CostModel(**calibrated)
    else:
        cost = CostModel(prefetch_mode=args.prefetch_mode, overlap_window_ms=args.overlap_window_ms)
    orders = document_orders(chunks)
    policy_factories = {
        "no_prefetch": lambda: NoPrefetchPolicy(),
        "random": lambda: RandomPolicy([c.chunk_id for c in chunks], args.seed),
        "sequential": lambda: SequentialPolicy(orders),
        "relative_offset_transition": lambda: RelativeOffsetTransitionPolicy.fit(
            splits["train"], orders
        ),
        "cosine": lambda: CosinePolicy(similarity),
        "graph_fixed": lambda: FixedGraphPolicy(adjacency),
        "adaptive_offline_global": lambda: LinearGraphPolicy(raw_edges, offline_weights),
        "adaptive_offline_query_type_oracle": lambda: QueryTypeOraclePolicy(raw_edges, query_weights, offline_weights),
        "adaptive_online_scratch": lambda: OnlineAdaptivePolicy(raw_edges, (0.5, 0.5), eta, "adaptive_online_scratch"),
        "adaptive_online_warm": lambda: OnlineAdaptivePolicy(raw_edges, offline_weights, eta, "adaptive_online_warm"),
        "one_step_target_oracle": lambda: OraclePolicy(),
    }
    result_paths = []
    for k in args.k:
        for policy_name, factory in policy_factories.items():
            logger.info("Running %s K=%d on %d test events", policy_name, k, len(test_events))
            result_path = output / f"results_{policy_name}_k{k}.csv"
            run_policy_trace(
                policy=factory(), events=test_events, chunks=chunks, k=k, cost_model=cost,
                capacity_entries=args.capacity_entries, capacity_bytes=bytes_capacity,
                output_csv=result_path,
                checkpoint_json=output / "checkpoints" / f"{policy_name}_k{k}.json",
                run_metadata=metadata,
                checkpoint_every=(1 if args.smoke or args.stress_smoke else args.checkpoint_every),
            )
            result_paths.append(result_path)
    frame = load_result_csvs(result_paths)
    write_analysis(frame, output)
    logger.info("Complete. Review %s and %s", output / "summary.csv", output / "run_manifest.json")


if __name__ == "__main__":
    main()
