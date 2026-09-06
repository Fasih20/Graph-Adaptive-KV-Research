"""End-to-end leakage-safe API answer-quality experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data import DATASETS, load_structured_sample, nodes_from_row
from .graph import (
    Node,
    Prediction,
    build_document_graph,
    cosine_matrix,
    cosine_query,
    observe_online,
    predict_adaptive,
    predict_cosine,
    predict_fixed,
    project_simplex,
    support_recall,
)
from .provider import GeminiClient, Generation, ProviderError, QuotaExhaustedError
from .scoring import exact_match, max_token_f1
from .stats import summarise


LOGGER = logging.getLogger("graphkv_quality")
POLICIES = ("cosine", "graph_fixed", "adaptive_offline_global", "adaptive_online_warm")
PROMPT_TEMPLATE = (
    "Answer the question using only the reference context. Return only the shortest answer "
    "that directly answers the question; do not explain.\n\n"
    "REFERENCE CONTEXT\n{context}\n\nQUESTION\n{question}\n\nANSWER"
)


@dataclass
class PreparedCase:
    row: dict
    nodes: list[Node]
    support_ids: set[int]
    similarity: np.ndarray
    adjacency: dict
    raw_edges: dict
    primary: int
    feedback_target: int


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _sha256(payload) -> str:
    serialised = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def prepare_case(row: dict, embedder, *, top_m: int, max_degree: int) -> PreparedCase:
    nodes, support_ids = nodes_from_row(row)
    if len(nodes) < 3 or not support_ids:
        raise ValueError(f"Question {row['id']} has insufficient nodes/support labels")
    encoded = embedder.encode(
        [row["question"], *[node.text for node in nodes]],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    query_embedding = encoded[0]
    embeddings = encoded[1:]
    similarity = cosine_matrix(embeddings)
    adjacency, raw_edges = build_document_graph(
        similarity,
        nodes,
        top_m=top_m,
        fixed_weights=(0.7, 0.3),
        max_degree=max_degree,
    )
    query_similarity = cosine_query(embeddings, query_embedding)
    primary = int(np.argmax(query_similarity))
    alternatives = sorted(support_ids - {primary})
    target_pool = alternatives or sorted(support_ids)
    feedback_target = max(target_pool, key=lambda node_id: (query_similarity[node_id], -node_id))
    return PreparedCase(row, nodes, support_ids, similarity, adjacency, raw_edges, primary, int(feedback_target))


def prepare_cases(rows: list[dict], embedder, *, top_m: int, max_degree: int, label: str) -> list[PreparedCase]:
    cases = []
    for index, row in enumerate(rows, start=1):
        cases.append(prepare_case(row, embedder, top_m=top_m, max_degree=max_degree))
        if index == 1 or index % 25 == 0 or index == len(rows):
            LOGGER.info("Prepared %s graph %d/%d", label, index, len(rows))
    return cases


def _selected(primary: int, prediction: Prediction) -> list[int]:
    result = [int(primary)]
    for node_id in prediction.ids:
        node_id = int(node_id)
        if node_id not in result:
            result.append(node_id)
    return result


def _predict(case: PreparedCase, policy: str, k: int, offline_weights, online_weights) -> Prediction:
    k = min(int(k), len(case.nodes) - 1)
    if policy == "cosine":
        return predict_cosine(case.similarity, case.primary, k)
    if policy == "graph_fixed":
        return predict_fixed(case.adjacency, case.primary, k)
    if policy == "adaptive_offline_global":
        return predict_adaptive(case.raw_edges, case.primary, k, offline_weights)
    if policy == "adaptive_online_warm":
        return predict_adaptive(case.raw_edges, case.primary, k, online_weights)
    raise ValueError(policy)


def fit_offline_weights(cases: list[PreparedCase], k_values: list[int], step: float = 0.05):
    divisions = int(round(1.0 / float(step)))
    rows = []
    for index in range(divisions + 1):
        alpha = index / divisions
        weights = (alpha, 1.0 - alpha)
        recalls = []
        target_hits = []
        for case in cases:
            for k in k_values:
                prediction = predict_adaptive(case.raw_edges, case.primary, k, weights)
                selected = _selected(case.primary, prediction)
                recalls.append(support_recall(selected, case.support_ids))
                target_hits.append(float(case.feedback_target in selected))
        rows.append(
            {
                "alpha": alpha,
                "beta": 1.0 - alpha,
                "train_mean_support_recall": float(np.mean(recalls)),
                "train_target_hit_rate": float(np.mean(target_hits)),
            }
        )
    best = max(
        rows,
        key=lambda row: (
            row["train_mean_support_recall"],
            row["train_target_hit_rate"],
            -abs(row["alpha"] - 0.7),
            -row["alpha"],
        ),
    )
    return np.asarray([best["alpha"], best["beta"]], dtype=np.float64), rows


def tune_online_eta(cases: list[PreparedCase], k_values: list[int], initial_weights):
    rows = []
    for eta in (0.01, 0.05, 0.1, 0.2, 0.5, 1.0):
        states = {int(k): project_simplex(initial_weights) for k in k_values}
        recalls = []
        target_hits = []
        updates = 0
        for case in cases:
            for k in k_values:
                prediction = predict_adaptive(case.raw_edges, case.primary, k, states[int(k)])
                selected = _selected(case.primary, prediction)
                recalls.append(support_recall(selected, case.support_ids))
                target_hits.append(float(case.feedback_target in selected))
                states[int(k)], update = observe_online(states[int(k)], case.feedback_target, prediction, eta)
                updates += int(update["updated"])
        rows.append(
            {
                "eta": eta,
                "dev_mean_support_recall": float(np.mean(recalls)),
                "dev_target_hit_rate": float(np.mean(target_hits)),
                "updates": updates,
            }
        )
    best = max(
        rows,
        key=lambda row: (
            row["dev_mean_support_recall"],
            row["dev_target_hit_rate"],
            -abs(row["eta"] - 0.1),
        ),
    )
    return float(best["eta"]), rows


def _context_text(case: PreparedCase, selected_ids: list[int]) -> str:
    return "\n\n".join(case.nodes[node_id].text for node_id in selected_ids)


def equal_reference_token_contexts(case, selected_by_policy, tokenizer, maximum_tokens: int):
    raw_ids = {
        policy: tokenizer(_context_text(case, ids), add_special_tokens=False)["input_ids"]
        for policy, ids in selected_by_policy.items()
    }
    target = min(int(maximum_tokens), *(len(ids) for ids in raw_ids.values()))
    if target <= 0:
        raise ValueError("No reference context tokens available")
    for _ in range(6):
        contexts = {
            policy: tokenizer.decode(ids[:target], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for policy, ids in raw_ids.items()
        }
        counts = {
            policy: len(tokenizer(text, add_special_tokens=False)["input_ids"])
            for policy, text in contexts.items()
        }
        if len(set(counts.values())) == 1:
            exact = next(iter(counts.values()))
            delivered = {}
            for policy, node_ids in selected_by_policy.items():
                delivered[policy] = []
                for prefix_size in range(1, len(node_ids) + 1):
                    prefix_tokens = tokenizer(
                        _context_text(case, node_ids[:prefix_size]), add_special_tokens=False
                    )["input_ids"]
                    if len(prefix_tokens) <= exact:
                        delivered[policy] = node_ids[:prefix_size]
                    else:
                        break
            return contexts, exact, delivered
        target = min(counts.values())
    raise RuntimeError(f"Could not equalise reference token budgets: {counts}")


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"events": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _generation_from_cache(value: dict) -> Generation:
    return Generation(
        text=value["text"],
        input_tokens=value.get("input_tokens"),
        output_tokens=value.get("output_tokens"),
        latency_ms=float(value.get("latency_ms", 0.0)),
        attempts=int(value.get("attempts", 1)),
        finish_reason=value.get("finish_reason"),
    )


def _save_outputs(output_dir: Path, events: list[dict], seed: int, *, print_table: bool = False) -> None:
    events = sorted(events, key=lambda row: (int(row["k"]), row["question_index"], row["policy"]))
    _write_csv(output_dir / "question_policy_results.csv", events)
    summary, deltas = summarise(events, seed=seed)
    _write_csv(output_dir / "summary.csv", summary)
    _write_csv(output_dir / "paired_deltas_vs_cosine.csv", deltas)
    if print_table:
        print("\nEXACT MEAN ANSWER-QUALITY RESULTS")
        print("policy                     K    N       EM       F1  support_recall  target_hit")
        print("-------------------------  --  ---  -------  -------  --------------  ----------")
        for row in summary:
            print(
                f"{row['policy']:<25}  {row['k']:>2}  {row['questions']:>3}  "
                f"{row['mean_em']:.4f}  {row['mean_f1']:.4f}  "
                f"{row['mean_support_recall']:.4f}          {row['target_hit_rate']:.4f}"
            )


def run(args) -> int:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "dataset_cache"
    scientific_config = {
        "version": "1.1.0",
        "dataset": args.dataset,
        "dataset_source": DATASETS[args.dataset]["repo"],
        "model": args.model,
        "embedding_model": args.embedding_model,
        "budget_tokenizer": args.budget_tokenizer,
        "n_train": args.n_train,
        "n_dev": args.n_dev,
        "n_test": args.n_test,
        "k_values": args.k_values,
        "top_m": args.top_m,
        "max_degree": args.max_degree,
        "context_token_budget": args.context_token_budget,
        "max_output_tokens": args.max_output_tokens,
        "seed": args.seed,
    }
    config_hash = _sha256(scientific_config)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("scientific_config_hash") != config_hash:
            raise RuntimeError(
                "Output directory belongs to a different scientific configuration. "
                "Use a new --output-dir or restore the original arguments."
            )
    else:
        _atomic_json(
            manifest_path,
            {
                "scientific_config_hash": config_hash,
                "scientific_config": scientific_config,
                "api_key_saved": False,
                "note": "RPM/retry/timeouts are operational and may change safely when resuming.",
            },
        )

    client = GeminiClient(
        args.model,
        api_key=args.api_key,
        rpm=args.rpm,
        max_retries=args.max_retries,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        max_retry_wait=args.max_retry_wait,
    )
    model_info = client.preflight()
    _atomic_json(output_dir / "gemini_model_info.json", model_info)
    LOGGER.info("Gemini preflight passed: %s", model_info.get("name"))
    if args.preflight_only:
        return 0
    complete_path = output_dir / "COMPLETE.json"
    state_path = output_dir / "checkpoint.json"
    if complete_path.exists() and state_path.exists():
        state = _load_state(state_path)
        LOGGER.info("Run is already complete; no dataset loading or API calls are needed")
        _save_outputs(output_dir, list(state["events"].values()), args.seed, print_table=True)
        return 0

    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    dataset_label = DATASETS[args.dataset]["label"]
    LOGGER.info("Loading structured %s samples", dataset_label)
    train_and_dev = load_structured_sample(
        args.dataset,
        "train", args.n_train + args.n_dev, args.seed, cache_dir, shuffle_buffer=args.shuffle_buffer
    )
    test_rows = load_structured_sample(
        args.dataset,
        "validation", args.n_test, args.seed + 1, cache_dir, shuffle_buffer=args.shuffle_buffer
    )
    train_rows = train_and_dev[: args.n_train]
    dev_rows = train_and_dev[args.n_train :]
    embedder = SentenceTransformer(args.embedding_model, device=args.embedding_device)
    tokenizer = AutoTokenizer.from_pretrained(args.budget_tokenizer, use_fast=True)
    train_cases = prepare_cases(train_rows, embedder, top_m=args.top_m, max_degree=args.max_degree, label="train")
    dev_cases = prepare_cases(dev_rows, embedder, top_m=args.top_m, max_degree=args.max_degree, label="dev")
    test_cases = prepare_cases(test_rows, embedder, top_m=args.top_m, max_degree=args.max_degree, label="test")

    offline_weights, weight_grid = fit_offline_weights(train_cases, args.k_values, step=args.weight_step)
    eta, eta_grid = tune_online_eta(dev_cases, args.k_values, offline_weights)
    fitted = {
        "offline_weights": {"semantic_alpha": float(offline_weights[0]), "structural_beta": float(offline_weights[1])},
        "online_eta": eta,
        "fit_label": f"{dataset_label} supporting-fact recall on disjoint training/dev samples; no API answers used",
        "weight_grid": weight_grid,
        "eta_grid": eta_grid,
    }
    _atomic_json(output_dir / "fitted_policy.json", fitted)
    LOGGER.info("Fitted offline weights alpha=%.2f beta=%.2f; online eta=%.2f", offline_weights[0], offline_weights[1], eta)

    cache_path = output_dir / "response_cache.json"
    state = _load_state(state_path)
    response_cache = _load_cache(cache_path)
    online_states = {int(k): offline_weights.copy() for k in args.k_values}
    completed_before = len(state["events"])
    if completed_before:
        LOGGER.info("Resuming: %d policy calls already checkpointed", completed_before)

    try:
        for question_index, case in enumerate(test_cases):
            for k in args.k_values:
                weights_before = online_states[int(k)].copy()
                predictions = {
                    policy: _predict(case, policy, k, offline_weights, weights_before)
                    for policy in POLICIES
                }
                selected = {policy: _selected(case.primary, prediction) for policy, prediction in predictions.items()}
                contexts, reference_tokens, delivered = equal_reference_token_contexts(
                    case, selected, tokenizer, args.context_token_budget
                )
                order = list(POLICIES)
                random.Random(args.seed + question_index * 1009 + int(k)).shuffle(order)
                for policy in order:
                    event_key = f"{case.row['id']}|K{k}|{policy}"
                    prompt = PROMPT_TEMPLATE.format(context=contexts[policy], question=case.row["question"])
                    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    if event_key in state["events"]:
                        if state["events"][event_key].get("prompt_hash") != prompt_hash:
                            raise RuntimeError(f"Resume mismatch for {event_key}; use a new output directory")
                        continue

                    cache_key = _sha256(
                        {"provider": "gemini", "model": args.model, "prompt_hash": prompt_hash, "max_output_tokens": args.max_output_tokens}
                    )
                    cache_hit = cache_key in response_cache
                    if cache_hit:
                        generation = _generation_from_cache(response_cache[cache_key])
                    else:
                        generation = client.generate(prompt, max_output_tokens=args.max_output_tokens)
                        response_cache[cache_key] = asdict(generation)
                        _atomic_json(cache_path, response_cache)

                    prediction = predictions[policy]
                    chosen = selected[policy]
                    delivered_ids = delivered[policy]
                    event = {
                        "status": "complete",
                        "question_index": question_index,
                        "question_id": case.row["id"],
                        "question": case.row["question"],
                        "gold_answers": json.dumps(case.row["answers"], ensure_ascii=False),
                        "k": int(k),
                        "policy": policy,
                        "model": args.model,
                        "answer": generation.text,
                        "em": exact_match(generation.text, case.row["answers"]),
                        "f1": max_token_f1(generation.text, case.row["answers"]),
                        "support_recall": support_recall(delivered_ids, case.support_ids),
                        "selection_support_recall_before_budget": support_recall(chosen, case.support_ids),
                        "target_hit": float(case.feedback_target in delivered_ids),
                        "selection_target_hit_before_budget": float(case.feedback_target in chosen),
                        "primary_node": case.primary,
                        "feedback_target_node": case.feedback_target,
                        "selected_node_ids": json.dumps(chosen),
                        "selected_nodes": len(chosen),
                        "fully_delivered_node_ids": json.dumps(delivered_ids),
                        "fully_delivered_nodes": len(delivered_ids),
                        "candidate_universe": len(prediction.candidate_universe),
                        "reference_context_tokens": reference_tokens,
                        "provider_input_tokens": generation.input_tokens,
                        "provider_output_tokens": generation.output_tokens,
                        "api_latency_ms": generation.latency_ms,
                        "api_attempts": generation.attempts,
                        "finish_reason": generation.finish_reason,
                        "response_cache_hit": cache_hit,
                        "prompt_hash": prompt_hash,
                        "online_weights_before": json.dumps(weights_before.tolist()) if policy == "adaptive_online_warm" else None,
                        "online_weights_after": None,
                        "online_updated": None,
                        "online_update_reason": None,
                    }
                    state["events"][event_key] = event
                    _atomic_json(state_path, state)
                    LOGGER.info(
                        "[%d/%d K=%d] %-24s EM=%.0f F1=%.3f%s",
                        question_index + 1,
                        len(test_cases),
                        k,
                        policy,
                        event["em"],
                        event["f1"],
                        " (response cache)" if cache_hit else "",
                    )

                online_prediction = predictions["adaptive_online_warm"]
                online_states[int(k)], update = observe_online(
                    weights_before, case.feedback_target, online_prediction, eta
                )
                online_key = f"{case.row['id']}|K{k}|adaptive_online_warm"
                online_event = state["events"][online_key]
                online_event["online_weights_after"] = json.dumps(update["weights_after"])
                online_event["online_updated"] = float(update["updated"])
                online_event["online_update_reason"] = update["update_reason"]
                _atomic_json(state_path, state)

            _save_outputs(output_dir, list(state["events"].values()), args.seed)
    except (QuotaExhaustedError, ProviderError, KeyboardInterrupt) as error:
        _save_outputs(output_dir, list(state["events"].values()), args.seed, print_table=True)
        LOGGER.error("Run stopped safely: %s", error)
        LOGGER.error("Progress is in %s; rerun the identical command to resume", state_path)
        return 2
    except Exception:
        _save_outputs(output_dir, list(state["events"].values()), args.seed, print_table=True)
        LOGGER.exception("Unexpected failure; completed API calls remain checkpointed in %s", state_path)
        raise

    events = list(state["events"].values())
    _save_outputs(output_dir, events, args.seed, print_table=True)
    expected = len(test_cases) * len(args.k_values) * len(POLICIES)
    if len(events) != expected:
        raise RuntimeError(f"Only {len(events)}/{expected} expected events are present")
    _atomic_json(
        complete_path,
        {"complete": True, "events": len(events), "questions": len(test_cases), "policies": list(POLICIES), "k_values": args.k_values},
    )
    LOGGER.info("Complete: %s", output_dir)
    return 0
