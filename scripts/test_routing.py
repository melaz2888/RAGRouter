#!/usr/bin/env python3
"""
Test similarity-based routing.

For each question:
1. Embed question
2. Get top similarity score from ChromaDB
3. If score > threshold → rag, else → direct
4. Compare to expected route
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retriever import embed_query, _get_collection


def get_top_similarity(question: str) -> float:
    """Get the highest similarity score between question and corpus."""
    query_embedding = embed_query(question)
    collection = _get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=["distances"],
    )

    distances = (results.get("distances") or [[]])[0]
    if not distances:
        return 0.0

    # Convert distance to similarity (cosine distance → similarity)
    distance = distances[0]
    similarity = 1.0 - distance
    return similarity


def route_by_similarity(question: str, threshold: float = 0.5) -> str:
    """Route based on similarity to corpus."""
    score = get_top_similarity(question)
    return "rag" if score >= threshold else "direct"


def test_routing(test_file: str, threshold: float = 0.5):
    """Test routing on a set of questions."""
    print(f"Testing with threshold: {threshold}")
    print("-" * 60)

    results = {"correct": 0, "incorrect": 0, "details": []}

    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            question = item["question"]
            expected = item["expected_route"]
            category = item.get("category", "unknown")

            score = get_top_similarity(question)
            predicted = "rag" if score >= threshold else "direct"
            correct = predicted == expected

            if correct:
                results["correct"] += 1
                status = "OK"
            else:
                results["incorrect"] += 1
                status = "WRONG"

            results["details"].append({
                "question": question[:50],
                "score": score,
                "predicted": predicted,
                "expected": expected,
                "correct": correct,
            })

            print(f"{status} [{category}] score={score:.3f} -> {predicted} (expected {expected})")
            print(f"   {question[:60]}...")

    print("-" * 60)
    total = results["correct"] + results["incorrect"]
    accuracy = results["correct"] / total * 100
    print(f"Accuracy: {results['correct']}/{total} ({accuracy:.1f}%)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold")
    parser.add_argument("--test-file", type=str, default="data/test_questions.jsonl")
    args = parser.parse_args()

    test_routing(args.test_file, args.threshold)
