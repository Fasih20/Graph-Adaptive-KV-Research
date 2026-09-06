"""
qa_scoring.py
==============
Standard SQuAD/HotpotQA-style normalization + Exact Match + token F1.
Pure Python, no API calls — this is both the free option AND the standard
metric this benchmark is actually scored with, so it's the right default,
not just the cheap one. An LLM-judge pass is deliberately NOT included here
to keep the eval's API usage bounded and predictable; add one externally if
you want a second opinion on top of EM/F1.
"""

import re
import string
from collections import Counter
from typing import List


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match(prediction: str, gold_answers: List[str]) -> float:
    norm_pred = normalize_answer(prediction)
    return float(any(norm_pred == normalize_answer(g) for g in gold_answers))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_f1(prediction: str, gold_answers: List[str]) -> float:
    return max((f1_score(prediction, g) for g in gold_answers), default=0.0)
