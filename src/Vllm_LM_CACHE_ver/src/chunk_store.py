"""
chunk_store.py
===============
Owns the corpus: loads a LongBench split, chunks it, embeds it, builds the
similarity matrix + graph, and generates the (pri, sec, q_type) trial pairs.

The dataset-loading and chunking logic (HF download -> zip extract -> concat
contexts up to 100k chars -> RecursiveCharacterTextSplitter) and the
embedding step (all-MiniLM-L6-v2) are ported from the canonical script's
run_dataset(), unchanged, since they determine the exact chunk boundaries
and embeddings that build_graph/generate_pairs are validated against. The
graph build and pair generation themselves are the untouched functions from
graph_algorithms.py.
"""

import os
import time
import zipfile
import tempfile
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
# Import from the submodule directly, NOT from the top-level
# langchain_text_splitters package. The package __init__ eagerly imports a
# sentence-transformers-based splitter variant we don't use, which in turn
# pulls in torchcodec (audio/video decoding) and its libnvrtc dependency —
# unrelated to plain text chunking, but enough to hard-crash the import on
# environments (e.g. this Kaggle image) where that shared lib is missing.
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import graph_algorithms as ga
from exp_config import (CHUNK_SIZE, CHUNK_OVERLAP, DATASET_CONFIGS, RANDOM_SEED,
                     N_PAIRS, GRAPH_CONSTRUCTION, TOP_M)

logger = logging.getLogger("chunk_store")


class ChunkStore:
    def __init__(self, dataset_key: str, graph_construction: str = GRAPH_CONSTRUCTION,
                 top_m: int = TOP_M, char_budget: int = 100_000):
        if dataset_key not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset_key {dataset_key!r}; choose from {list(DATASET_CONFIGS)}")
        self.dataset_key = dataset_key
        self.cfg = DATASET_CONFIGS[dataset_key]
        self.graph_construction = graph_construction
        self.top_m = top_m
        self.char_budget = char_budget

        self.chunk_texts: List[str] = []
        self.embeddings: np.ndarray = None
        self.sim_matrix: np.ndarray = None
        self.adj_matrix = None
        self.raw_edges: dict = None
        self.embed_time_s: float = 0.0
        self.graph_build_time_s: float = 0.0

    # ── Loading / chunking (ported logic from run_dataset) ────────────────
    def load(self):
        cfg = self.cfg
        logger.info(f"DATASET: {cfg['label']}")
        t0 = time.perf_counter()
        zip_path = hf_hub_download(repo_id=cfg["hf_name"], filename="data.zip", repo_type="dataset")
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, "r") as z:
                target = f"data/{cfg['split']}.jsonl"
                z.extract(target, tmpdir)
                df_raw = pd.read_json(os.path.join(tmpdir, target), lines=True)

        contexts = []
        for ctx in df_raw[cfg["text_key"]]:
            contexts.append(ctx)
            if sum(len(c) for c in contexts) >= self.char_budget:
                break
        raw = "\n\n".join(contexts)
        logger.info(f"  Raw text: {len(raw):,} chars  ({time.perf_counter()-t0:.1f}s)")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len)
        self.chunk_texts = splitter.split_text(raw)
        logger.info(f"  Chunks: {len(self.chunk_texts)}")
        return self

    def embed(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info("Computing embeddings ...")
        t0 = time.perf_counter()
        embedder = SentenceTransformer(model_name)
        self.embeddings = embedder.encode(
            self.chunk_texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        self.embed_time_s = time.perf_counter() - t0
        logger.info(f"  Embeddings: {self.embeddings.shape}  ({self.embed_time_s:.2f}s)")
        self.sim_matrix = cosine_similarity(self.embeddings)
        return self

    def build_graph(self):
        self.adj_matrix, self.raw_edges, self.graph_build_time_s = ga.build_graph(
            self.embeddings, self.sim_matrix, logger,
            graph_construction=self.graph_construction, top_m=self.top_m)
        return self

    def generate_pairs(self, n_pairs: int = N_PAIRS, seed: int = RANDOM_SEED) -> List[Tuple[int, int, str]]:
        return ga.generate_pairs(len(self.chunk_texts), n_pairs, self.sim_matrix, seed, logger)

    def prepare(self):
        """Convenience: load -> embed -> build_graph, in the same order the
        canonical run_dataset() does it (embeddings before graph, since the
        graph needs sim_matrix derived from embeddings)."""
        self.load()
        self.embed()
        self.build_graph()
        return self

    # ── Accessors used by cache_manager.py ─────────────────────────────────
    def text(self, chunk_id: int) -> str:
        return self.chunk_texts[chunk_id]

    def texts(self, chunk_ids: List[int]) -> List[str]:
        return [self.chunk_texts[i] for i in chunk_ids]

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_texts)
