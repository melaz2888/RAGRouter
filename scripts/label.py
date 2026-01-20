#!/usr/bin/env python3
"""
Generate training labels by testing LLM accuracy.

Logic:
1. Load dataset with questions and answers
2. For each question, ask LLM (no context)
3. Compare LLM answer to real answer
4. If correct → "direct", if wrong → "rag"
5. Save labels to data/labels.jsonl

Usage:
    python scripts/label.py --dataset nq --limit 1000
    python scripts/label.py --input data/qa/questions.jsonl
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generator import generate_direct
from src.config import LABELS_FILE


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    import re
    # Lowercase
    s = s.lower()
    # Remove punctuation
    s = re.sub(r"[^\w\s]", "", s)
    # Remove extra whitespace
    s = " ".join(s.split())
    return s


def answers_match(predicted: str, expected: str, threshold: float = 0.5) -> bool:
    """
    Check if predicted answer matches expected answer.

    Uses simple token overlap. Returns True if significant overlap.
    """
    pred_tokens = set(normalize_answer(predicted).split())
    exp_tokens = set(normalize_answer(expected).split())

    if not exp_tokens:
        return False

    overlap = len(pred_tokens & exp_tokens)
    recall = overlap / len(exp_tokens)

    return recall >= threshold


def load_natural_questions(limit: int = None) -> List[Dict]:
    """
    Load Natural Questions dataset from Hugging Face.

    Returns list of {"question": str, "answer": str}
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[label] Please install datasets: pip install datasets")
        sys.exit(1)

    print("[label] Loading Natural Questions dataset from Hugging Face...")
    ds = load_dataset("sentence-transformers/natural-questions", split="train")

    items = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break

        question = row.get("query") or row.get("question", "")
        answer = row.get("answer") or ""

        if question and answer:
            items.append({"question": question, "answer": answer})

    print(f"[label] Loaded {len(items)} questions")
    return items


def load_jsonl(path: str) -> List[Dict]:
    """Load questions from a JSONL file."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                if obj.get("question"):
                    items.append(obj)
            except json.JSONDecodeError:
                continue
    return items


def generate_labels(
    items: List[Dict],
    output_path: str,
    skip_existing: bool = True,
) -> None:
    """
    Generate labels by testing LLM on each question.

    Args:
        items: List of {"question": str, "answer": str}
        output_path: Path to save labels.jsonl
        skip_existing: Skip questions if output file exists
    """
    # Load existing labels to skip
    existing = set()
    if skip_existing and Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing.add(obj.get("question", ""))
                except:
                    pass
        print(f"[label] Found {len(existing)} existing labels, will skip those")

    # Open output file in append mode
    mode = "a" if skip_existing and existing else "w"
    outfile = open(output_path, mode, encoding="utf-8")

    stats = {"direct": 0, "rag": 0, "errors": 0}

    for i, item in enumerate(items):
        question = item["question"]
        expected_answer = item.get("answer", "")

        # Skip if already labeled
        if question in existing:
            continue

        try:
            # Ask LLM without context
            result = generate_direct(question)
            predicted = result.answer

            # Compare answers
            if answers_match(predicted, expected_answer):
                label = "direct"
                stats["direct"] += 1
            else:
                label = "rag"
                stats["rag"] += 1

            # Save
            record = {
                "question": question,
                "answer": expected_answer,
                "route": label,
                "llm_answer": predicted,
            }
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            outfile.flush()

            # Progress
            if (i + 1) % 10 == 0:
                total = stats["direct"] + stats["rag"]
                print(f"[label] Processed {total}: {stats['direct']} direct, {stats['rag']} rag")

        except Exception as e:
            print(f"[label] Error on question {i}: {e}")
            stats["errors"] += 1

    outfile.close()

    print(f"\n[label] Done!")
    print(f"[label] Direct: {stats['direct']}")
    print(f"[label] RAG: {stats['rag']}")
    print(f"[label] Errors: {stats['errors']}")
    print(f"[label] Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate training labels")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["nq"],
        help="Dataset to use: nq (Natural Questions)",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to custom JSONL file with questions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(LABELS_FILE),
        help=f"Output path for labels (default: {LABELS_FILE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum questions to process (default: 500)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip existing labels, overwrite",
    )
    args = parser.parse_args()

    # Load questions
    if args.dataset == "nq":
        items = load_natural_questions(limit=args.limit)
    elif args.input:
        items = load_jsonl(args.input)
        if args.limit:
            items = items[:args.limit]
    else:
        print("[label] Please specify --dataset nq or --input <file>")
        sys.exit(1)

    if not items:
        print("[label] No questions loaded")
        sys.exit(1)

    # Generate labels
    generate_labels(
        items=items,
        output_path=args.output,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
