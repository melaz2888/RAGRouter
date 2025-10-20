"""
CPU-only ingestion utilities for RAGRouter.

Usage (notes):
- chunk_text(): sliding-window token chunks using tiktoken (cl100k_base), default L=512/S=128.
- count_tokens(): fast token count with the same tokenizer to keep consistency.
- load_corpus(): recursively load *.txt with UTF-8 (errors="ignore"), returning (doc_id, raw_text, {"source": abs_path}).
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple

import tiktoken


# Single tokenizer instance (cl100k_base) reused across functions
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Count tokens in `text` with cl100k_base.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return len(_ENCODING.encode(text))


def _sliding_indices(n: int, L: int, S: int) -> List[Tuple[int, int]]:
    """
    Produce (start, end) token-index windows for length L and stride S over a sequence of length n.
    Ensures at least one window if 0 < n < L.
    """
    if L <= 0:
        raise ValueError("L must be > 0")
    if S <= 0:
        raise ValueError("S must be > 0")

    if n == 0:
        return []

    idxs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        j = min(i + L, n)
        if j <= i:
            break
        idxs.append((i, j))
        if j == n:  # last window reaches the end
            break
        i += S

    # If text shorter than L and we produced nothing, emit single window
    if not idxs and n > 0:
        idxs.append((0, n))
    return idxs


def chunk_text(text: str, L: int = 512, S: int = 128) -> List[str]:
    """
    Chunk `text` into token windows using tiktoken (cl100k_base).
    - L: window length in tokens
    - S: stride in tokens
    Returns a list of NON-EMPTY chunk strings (decoded from token windows).
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.strip()
    if not text:
        return []

    tokens = _ENCODING.encode(text)
    windows = _sliding_indices(len(tokens), L, S)
    chunks: List[str] = []

    for start, end in windows:
        piece_tokens = tokens[start:end]
        if not piece_tokens:
            continue
        chunk = _ENCODING.decode(piece_tokens).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def load_corpus(corpus_dir: str) -> List[Tuple[str, str, Dict[str, str]]]:
    """
    Load all *.txt files under `corpus_dir` recursively.

    Returns:
      List of tuples: (doc_id, raw_text, {"source": absolute_path})
      - doc_id is the path RELATIVE to `corpus_dir` using POSIX slashes.
      - raw_text is read with UTF-8 and errors="ignore".
      - metadata["source"] is the absolute path to the file.
    Unreadable files are skipped (warning printed), processing continues.
    """
    if not corpus_dir or not os.path.isdir(corpus_dir):
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    pattern = os.path.join(corpus_dir, "**", "*.txt")
    paths = sorted(glob.glob(pattern, recursive=True))
    out: List[Tuple[str, str, Dict[str, str]]] = []

    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            print(f"[warn] skipping unreadable file: {p} ({e})")
            continue

        rel_id = os.path.relpath(p, corpus_dir).replace("\\", "/")
        meta = {"source": os.path.abspath(p)}
        out.append((rel_id, content, meta))

    return out
