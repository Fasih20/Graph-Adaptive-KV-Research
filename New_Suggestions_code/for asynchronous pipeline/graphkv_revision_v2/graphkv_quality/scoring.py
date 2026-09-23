"""Standard HotpotQA/SQuAD exact-match and token-F1 metrics."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold_answers: Sequence[str]) -> float:
    prediction = normalize_answer(prediction)
    return float(any(prediction == normalize_answer(answer) for answer in gold_answers))


def token_f1(prediction: str, gold: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(gold).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = Counter(predicted) & Counter(expected)
    same = sum(overlap.values())
    if same == 0:
        return 0.0
    precision = same / len(predicted)
    recall = same / len(expected)
    return 2.0 * precision * recall / (precision + recall)


def max_token_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    return max((token_f1(prediction, answer) for answer in gold_answers), default=0.0)
