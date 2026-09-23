"""Document-aware corpus, graph, and leakage-safe workload preparation."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import tempfile
import time
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

import graph_algorithms as legacy
from adaptive_policies import TraceEvent
from document_graph import build_graph_document_aware
from exp_config import (
    DATASET_CONFIGS,
    EMBEDDING_DEVICE,
    GRAPH_CONSTRUCTION,
    RANDOM_SEED,
    TOP_M,
)
from token_chunking import split_text_by_tokens
logger = logging.getLogger("chunk_store")


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: int
    document_id: str
    position: int
    text: str
    token_count: int
    kv_bytes: int


def model_kv_bytes(config, token_count: int, bytes_per_element: int = 2) -> int:
    # Text-only configs expose these fields directly. Composite multimodal
    # configs such as Gemma 3 4B keep the causal decoder under text_config.
    text_config = getattr(config, "text_config", config)
    heads = int(text_config.num_attention_heads)
    hidden = int(text_config.hidden_size)
    kv_heads = int(getattr(text_config, "num_key_value_heads", heads))
    head_dim = int(getattr(text_config, "head_dim", hidden // heads))
    return int(
        2
        * int(text_config.num_hidden_layers)
        * kv_heads
        * head_dim
        * int(token_count)
        * int(bytes_per_element)
    )


def split_document_ids(document_ids: Iterable[str], seed: int = RANDOM_SEED):
    ids = sorted(set(map(str, document_ids)))
    if len(ids) < 5:
        raise ValueError("at least five documents are required for train/dev/test separation")
    rng = np.random.default_rng(seed)
    shuffled = [ids[index] for index in rng.permutation(len(ids))]
    train_end = max(1, int(0.6 * len(shuffled)))
    dev_end = max(train_end + 1, int(0.8 * len(shuffled)))
    dev_end = min(dev_end, len(shuffled) - 1)
    return {
        "train": set(shuffled[:train_end]),
        "dev": set(shuffled[train_end:dev_end]),
        "test": set(shuffled[dev_end:]),
    }


def _exact_labels(count: int) -> list[str]:
    counts = {
        "semantic": count // 4,
        "structural": count // 4,
        "multi-hop": count - 2 * (count // 4),
    }
    return [name for name in ("semantic", "structural", "multi-hop") for _ in range(counts[name])]


class ChunkStore:
    def __init__(
        self,
        dataset_key: str,
        *,
        tokenizer,
        model_config,
        graph_construction: str = GRAPH_CONSTRUCTION,
        top_m: int = TOP_M,
        max_documents: int = 30,
        max_chars_per_document: int = 4000,
        document_chunk_tokens: int = 384,
        document_chunk_overlap_tokens: int = 64,
        embedding_device: str = EMBEDDING_DEVICE,
        seed: int = RANDOM_SEED,
    ):
        if dataset_key not in DATASET_CONFIGS:
            raise ValueError(f"unknown dataset {dataset_key!r}")
        self.dataset_key = dataset_key
        self.cfg = DATASET_CONFIGS[dataset_key]
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.graph_construction = graph_construction
        self.top_m = int(top_m)
        self.max_documents = int(max_documents)
        self.max_chars_per_document = int(max_chars_per_document)
        self.document_chunk_tokens = int(document_chunk_tokens)
        self.document_chunk_overlap_tokens = int(document_chunk_overlap_tokens)
        self.embedding_device = embedding_device
        self.seed = int(seed)
        self.chunks: list[ChunkRecord] = []
        self.chunk_texts: list[str] = []
        self.embeddings = None
        self.sim_matrix = None
        self.adj_matrix = None
        self.raw_edges = None
        self.embed_time_s = 0.0
        self.graph_build_time_s = 0.0
        self.document_splits: dict[str, set[str]] = {}

    def load(self):
        cfg = self.cfg
        logger.info("Loading %s with document boundaries preserved", cfg["label"])
        archive = hf_hub_download(repo_id=cfg["hf_name"], filename="data.zip", repo_type="dataset")
        with tempfile.TemporaryDirectory() as temporary:
            with zipfile.ZipFile(archive) as zipped:
                member = f"data/{cfg['split']}.jsonl"
                zipped.extract(member, temporary)
                frame = pd.read_json(os.path.join(temporary, member), lines=True)

        rng = np.random.default_rng(self.seed)
        count = min(self.max_documents, len(frame))
        selected = sorted(rng.choice(len(frame), size=count, replace=False).tolist())
        chunks: list[ChunkRecord] = []
        for row_index in selected:
            document_id = f"{self.dataset_key}:{row_index}"
            text = str(frame.iloc[row_index][cfg["text_key"]])[: self.max_chars_per_document]
            document_chunks = split_text_by_tokens(
                self.tokenizer,
                text,
                self.document_chunk_tokens,
                self.document_chunk_overlap_tokens,
            )
            for position, chunk_text in enumerate(document_chunks):
                token_count = len(self.tokenizer.encode(chunk_text, add_special_tokens=False))
                chunks.append(
                    ChunkRecord(
                        chunk_id=len(chunks),
                        document_id=document_id,
                        position=position,
                        text=chunk_text,
                        token_count=token_count,
                        kv_bytes=model_kv_bytes(self.model_config, token_count),
                    )
                )
        if len(chunks) < 20:
            raise RuntimeError("corpus produced too few chunks")
        self.chunks = chunks
        self.chunk_texts = [chunk.text for chunk in chunks]
        self.document_splits = split_document_ids(
            [chunk.document_id for chunk in chunks], self.seed
        )
        logger.info("Prepared %d chunks from %d documents", len(chunks), count)
        return self

    def embed(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        logger.info("Computing embeddings on %s", self.embedding_device)
        started = time.perf_counter()
        embedder = SentenceTransformer(model_name, device=self.embedding_device)
        self.embeddings = embedder.encode(
            self.chunk_texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.embed_time_s = time.perf_counter() - started
        values = np.asarray(self.embeddings, dtype=np.float32)
        self.sim_matrix = values @ values.T
        return self

    def build_graph(self):
        self.adj_matrix, self.raw_edges, self.graph_build_time_s = build_graph_document_aware(
            self.embeddings,
            self.sim_matrix,
            [chunk.document_id for chunk in self.chunks],
            [chunk.position for chunk in self.chunks],
            logger,
            self.graph_construction,
            self.top_m,
        )
        return self

    def prepare(self):
        return self.load().embed().build_graph()

    def workload_fingerprint(self) -> str:
        """Stable identity used to prevent reuse of mismatched sanity data."""
        digest = hashlib.sha256()
        for chunk in self.chunks:
            digest.update(
                json.dumps(asdict(chunk), sort_keys=True, ensure_ascii=False).encode("utf-8")
            )
            digest.update(b"\n")
        return digest.hexdigest()

    def generate_events(self, split: str, count: int, seed_offset: int = 0) -> list[TraceEvent]:
        allowed = self.document_splits[split]
        grouped: dict[str, list[ChunkRecord]] = defaultdict(list)
        for chunk in self.chunks:
            if chunk.document_id in allowed:
                grouped[chunk.document_id].append(chunk)
        grouped = {
            document: sorted(values, key=lambda chunk: chunk.position)
            for document, values in grouped.items()
            if len(values) >= 4
        }
        if not grouped:
            raise RuntimeError(f"no usable documents in {split} split")
        rng = np.random.default_rng(self.seed + seed_offset)
        labels = _exact_labels(int(count))
        rng.shuffle(labels)
        documents = sorted(grouped)
        document_cycle = [documents[index % len(documents)] for index in range(count)]
        rng.shuffle(document_cycle)
        steps = defaultdict(int)
        events = []
        for event_id, (query_type, document_id) in enumerate(zip(labels, document_cycle)):
            ordered = grouped[document_id]
            current_index = int(rng.integers(0, max(1, len(ordered) - 3)))
            current = ordered[current_index]
            if query_type == "structural":
                target = ordered[min(len(ordered) - 1, current_index + int(rng.choice([1, 2, 3])))]
            else:
                eligible = [
                    candidate
                    for index, candidate in enumerate(ordered)
                    if abs(index - current_index) > 2
                ] or [ordered[min(len(ordered) - 1, current_index + 1)]]
                semantic = max(
                    eligible,
                    key=lambda candidate: (
                        self.sim_matrix[current.chunk_id, candidate.chunk_id],
                        -candidate.chunk_id,
                    ),
                )
                if query_type == "multi-hop" and rng.random() < 0.5:
                    target = ordered[min(len(ordered) - 1, current_index + 2)]
                else:
                    target = semantic
            events.append(
                TraceEvent(
                    event_id=event_id,
                    session_id=f"{split}:{document_id}",
                    step_id=steps[document_id],
                    document_id=document_id,
                    current_chunk_id=current.chunk_id,
                    target_chunk_id=target.chunk_id,
                    query_type=query_type,
                )
            )
            steps[document_id] += 1
        return events

    def generate_pairs(self, n_pairs: int, seed: int = RANDOM_SEED):
        """Legacy compatibility only; new runs use ``generate_events``."""
        return legacy.generate_pairs(len(self.chunks), n_pairs, self.sim_matrix, seed, logger)

    def save(self, output_dir: Path, traces: dict[str, list[TraceEvent]] | None = None):
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        if traces:
            fields = list(TraceEvent.__dataclass_fields__)
            for name, events in traces.items():
                with (output_dir / f"trace_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(asdict(event) for event in events)

    def text(self, chunk_id: int) -> str:
        return self.chunks[int(chunk_id)].text

    def texts(self, chunk_ids: List[int]) -> List[str]:
        return [self.text(chunk_id) for chunk_id in chunk_ids]

    def chunk(self, chunk_id: int) -> ChunkRecord:
        return self.chunks[int(chunk_id)]

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)
