"""
text_chunking.py
=================
Provides get_recursive_character_text_splitter(), which returns a working
RecursiveCharacterTextSplitter-compatible class: the real one from
langchain_text_splitters if it imports cleanly, otherwise a small
dependency-free local reimplementation of the same chunk_size /
chunk_overlap / separator-hierarchy behavior.

WHY THIS EXISTS (read before "simplifying" this back to a plain import):

langchain_text_splitters/__init__.py unconditionally does
`from langchain_text_splitters.sentence_transformers import
SentenceTransformersTokenTextSplitter` — and THAT submodule does
`from sentence_transformers import SentenceTransformer` (the external
package, confusingly same-ish name) guarded only by `except ImportError`.
On some images (Kaggle's default image, observed in practice) that
external import cascades into torchcodec -> libnvrtc and fails with
something other than a plain ImportError, which isn't caught, so
__init__.py itself raises.

Critically: importing the submodule directly —
`from langchain_text_splitters.character import RecursiveCharacterTextSplitter`
— does NOT route around this. Python always executes a package's
__init__.py before any of its submodules, regardless of which name you're
importing; there's no way to reach `.character` without `__init__.py`
running first. (An earlier pass at this fix assumed submodule-direct
import would avoid the parent __init__.py — confirmed against
langchain's actual current __init__.py source that this assumption was
wrong; the earlier "fix" doesn't reliably do anything.)

So: try the real import, and only fall back if it actually fails, for
whatever reason. The fallback isn't a byte-for-byte port of langchain's
regex/keep_separator edge cases, just a solid splitter with the same
chunk_size / chunk_overlap / separator-hierarchy contract — sufficient for
this project's actual use (chunking concatenated article text and QA
context strings; nothing exotic like markdown or code-language
separators is used here). If you're comparing chunk boundaries against a
run that used the real library, note in your results which one actually
ran — logged clearly either way.
"""

import logging

logger = logging.getLogger(__name__)

_real_splitter_cls = None
_fallback_reason = None
try:
    from langchain_text_splitters.character import RecursiveCharacterTextSplitter as _real_splitter_cls
except Exception as e:  # noqa: BLE001 - deliberately broad, see module docstring
    _fallback_reason = f"{type(e).__name__}: {e}"


class _FallbackRecursiveCharacterTextSplitter:
    """Local reimplementation, used only if the real import above failed."""

    def __init__(self, chunk_size=4000, chunk_overlap=200, length_function=len,
                separators=None, **_ignored_kwargs):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separators = separators if separators is not None else ["\n\n", "\n", " ", ""]

    def _merge_splits(self, splits, separator):
        sep_len = self.length_function(separator)
        chunks, current, current_len = [], [], 0
        for piece in splits:
            piece_len = self.length_function(piece)
            added_len = piece_len + (sep_len if current else 0)
            if current and current_len + added_len > self.chunk_size:
                chunk = separator.join(current)
                if chunk:
                    chunks.append(chunk)
                overlap, overlap_len = [], 0
                for p in reversed(current):
                    p_len = self.length_function(p) + (sep_len if overlap else 0)
                    if overlap_len + p_len > self.chunk_overlap:
                        break
                    overlap.insert(0, p)
                    overlap_len += p_len
                current, current_len = overlap, overlap_len
                added_len = piece_len + (sep_len if current else 0)
            current.append(piece)
            current_len += added_len
        if current:
            chunk = separator.join(current)
            if chunk:
                chunks.append(chunk)
        return chunks

    def _split_text(self, text, separators):
        separator, remaining = separators[-1], []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator, remaining = sep, separators[i + 1:]
                break
        splits = list(text) if separator == "" else text.split(separator)

        good_splits, final_chunks = [], []
        for s in splits:
            if self.length_function(s) < self.chunk_size:
                good_splits.append(s)
                continue
            if good_splits:
                final_chunks.extend(self._merge_splits(good_splits, separator))
                good_splits = []
            final_chunks.extend(self._split_text(s, remaining) if remaining else [s])
        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator))
        return final_chunks

    def split_text(self, text):
        if not text:
            return []
        return self._split_text(text, self.separators)


def get_recursive_character_text_splitter():
    """Returns the class to instantiate — call this instead of importing
    RecursiveCharacterTextSplitter directly anywhere in this project."""
    if _real_splitter_cls is not None:
        return _real_splitter_cls
    logger.warning(
        f"langchain_text_splitters import failed ({_fallback_reason}) — "
        f"using text_chunking.py's local fallback splitter instead. Chunk "
        f"boundaries may differ slightly from a run where the real library "
        f"loaded; this does not affect chunk_size/chunk_overlap targets.")
    return _FallbackRecursiveCharacterTextSplitter
