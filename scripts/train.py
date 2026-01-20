#!/usr/bin/env python3
"""
Train the router classifier.

Usage:
    python scripts/train.py
    python scripts/train.py --labels data/labels.jsonl
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.router import train
from src.config import LABELS_FILE


def load_labels(path: str) -> Tuple[List[str], List[str]]:
    """
    Load labels from JSONL file.

    Returns:
        (questions, labels) tuple
    """
    questions = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                question = obj.get("question", "")
                route = obj.get("route", "")

                if question and route in ("direct", "rag"):
                    questions.append(question)
                    labels.append(route)
            except json.JSONDecodeError:
                continue

    return questions, labels


def main():
    parser = argparse.ArgumentParser(description="Train the router classifier")
    parser.add_argument(
        "--labels",
        type=str,
        default=str(LABELS_FILE),
        help=f"Path to labels JSONL file (default: {LABELS_FILE})",
    )
    args = parser.parse_args()

    # Check file exists
    if not Path(args.labels).exists():
        print(f"[train] Labels file not found: {args.labels}")
        print("[train] Run 'python scripts/label.py' first to generate labels")
        sys.exit(1)

    # Load labels
    print(f"[train] Loading labels from {args.labels}")
    questions, labels = load_labels(args.labels)

    if not questions:
        print("[train] No valid labels found")
        sys.exit(1)

    print(f"[train] Loaded {len(questions)} labeled questions")
    print(f"[train] Direct: {labels.count('direct')}, RAG: {labels.count('rag')}")

    # Train
    metrics = train(questions, labels, save=True)

    print("\n[train] Training complete!")
    print(f"[train] Metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
