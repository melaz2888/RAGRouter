#!/usr/bin/env python3
"""
Ingest documents into ChromaDB.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --corpus data/corpus/synth
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retriever import ingest_documents
from src.config import CORPUS_DIR


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument(
        "--corpus",
        type=str,
        default=str(CORPUS_DIR),
        help=f"Directory containing .txt files (default: {CORPUS_DIR})",
    )
    args = parser.parse_args()

    print(f"[ingest] Starting ingestion from {args.corpus}")
    result = ingest_documents(args.corpus)
    print(f"[ingest] Result: {result}")


if __name__ == "__main__":
    main()
