"""Load structured multi-hop QA rows and preserve document boundaries."""

from __future__ import annotations

import json
from pathlib import Path

from .graph import Node


DATASETS = {
    "hotpot": {
        "repo": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "label": "HotpotQA distractor",
    },
    "2wiki": {
        "repo": "framolfese/2WikiMultihopQA",
        "config": None,
        "label": "2WikiMultiHopQA",
    },
}


def _as_lists(value, field: str) -> list:
    """Handle Hugging Face struct-of-lists and occasional list-of-struct forms."""
    if isinstance(value, dict):
        return list(value.get(field, []))
    if isinstance(value, list):
        return [item.get(field) for item in value]
    return []


def normalise_hotpot_row(row: dict) -> dict:
    context = row.get("context", {})
    titles = _as_lists(context, "title")
    sentence_groups = _as_lists(context, "sentences")
    if not titles and isinstance(context, list):
        titles = [item.get("title") for item in context]
        sentence_groups = [item.get("sentences", []) for item in context]

    supporting = row.get("supporting_facts", {})
    support_titles = _as_lists(supporting, "title")
    support_indices = _as_lists(supporting, "sent_id")
    support_pairs = {
        (str(title), int(position))
        for title, position in zip(support_titles, support_indices)
        if title is not None and position is not None
    }
    documents = []
    for title, sentences in zip(titles, sentence_groups):
        documents.append({"title": str(title), "sentences": [str(sentence) for sentence in sentences]})
    answer = row.get("answer", "")
    answers = [str(item) for item in answer] if isinstance(answer, list) else [str(answer)]
    return {
        "id": str(row.get("id", row.get("_id", ""))),
        "question": str(row.get("question", row.get("input", ""))),
        "answers": answers,
        "documents": documents,
        "support_pairs": sorted([list(pair) for pair in support_pairs]),
        "type": row.get("type"),
        "level": row.get("level"),
    }


def nodes_from_row(row: dict) -> tuple[list[Node], set[int]]:
    support_pairs = {(str(title), int(position)) for title, position in row["support_pairs"]}
    nodes: list[Node] = []
    support_ids: set[int] = set()
    for document in row["documents"]:
        title = str(document["title"])
        for position, sentence in enumerate(document["sentences"]):
            is_support = (title, position) in support_pairs
            node_id = len(nodes)
            text = f"[{title}] {str(sentence).strip()}"
            nodes.append(Node(node_id, title, position, text, is_support))
            if is_support:
                support_ids.add(node_id)
    return nodes, support_ids


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_structured_sample(
    dataset: str,
    split: str,
    count: int,
    seed: int,
    cache_dir: Path,
    *,
    shuffle_buffer: int = 10_000,
) -> list[dict]:
    """Stream a deterministic sample, cache it, and never redownload on resume."""
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported structured dataset: {dataset!r}")
    spec = DATASETS[dataset]
    cache_path = cache_dir / f"{dataset}_{split}_n{count}_seed{seed}.jsonl"
    if cache_path.exists():
        rows = _read_jsonl(cache_path)
        if len(rows) != count:
            raise RuntimeError(f"Dataset cache {cache_path} has {len(rows)} rows, expected {count}")
        return rows

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Install requirements.txt; the 'datasets' package is required") from error

    load_args = [spec["repo"]]
    if spec["config"] is not None:
        load_args.append(spec["config"])
    stream = load_dataset(*load_args, split=split, streaming=True)
    stream = stream.shuffle(seed=int(seed), buffer_size=int(shuffle_buffer))
    rows = []
    for raw in stream:
        row = normalise_hotpot_row(dict(raw))
        if row["question"] and row["answers"] and row["documents"] and row["support_pairs"]:
            rows.append(row)
        if len(rows) >= count:
            break
    if len(rows) != count:
        raise RuntimeError(f"Only found {len(rows)} usable {split} rows; requested {count}")
    _write_jsonl(cache_path, rows)
    return rows
