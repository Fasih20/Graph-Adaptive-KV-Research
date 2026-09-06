from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np


DATASETS = {
    "hotpot": ("hotpotqa", "context"),
    "2wiki": ("2wikimqa", "context"),
    "musique": ("musique", "context"),
    "multifield": ("multifieldqa_en", "context"),
}


def load_longbench_documents(
    dataset: str,
    max_documents: int,
    seed: int = 42,
    max_chars_per_document: int | None = None,
) -> dict[str, str]:
    """Download LongBench and retain record/document boundaries."""

    if dataset not in DATASETS:
        raise ValueError(f"unknown LongBench dataset {dataset!r}")
    from huggingface_hub import hf_hub_download

    split, text_key = DATASETS[dataset]
    archive = hf_hub_download(
        repo_id="THUDM/LongBench", filename="data.zip", repo_type="dataset"
    )
    with zipfile.ZipFile(archive) as zipped:
        member = f"data/{split}.jsonl"
        with zipped.open(member) as handle:
            records = [json.loads(line) for line in handle]
    rng = np.random.default_rng(seed)
    count = min(int(max_documents), len(records))
    selected = sorted(rng.choice(len(records), size=count, replace=False).tolist())
    documents = {}
    for index in selected:
        text = str(records[index][text_key])
        if max_chars_per_document is not None:
            text = text[: int(max_chars_per_document)]
        documents[f"{dataset}:{index}"] = text
    return documents

