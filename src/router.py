"""
RAGRouter v2 Router

The "meta-model" that decides: direct or rag?

Routing logic:
1. Check keywords first → if match, return "rag"
2. Use classifier on query-only features → predict route

Key principle: NO retrieval calls during routing.
All features are derived from the question text only.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Literal

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .config import (
    KEYWORDS_FILE,
    ROUTER_MODEL_PATH,
    TFIDF_MODEL_PATH,
    MODELS_DIR,
    DEFAULT_ROUTE,
    ROUTER_TRAIN_TEST_SPLIT,
    ROUTER_RANDOM_SEED,
)


Route = Literal["direct", "rag"]


# =============================================================================
# KEYWORDS
# =============================================================================

_keywords: List[str] | None = None


def _load_keywords() -> List[str]:
    """Load domain keywords from file."""
    global _keywords
    if _keywords is not None:
        return _keywords

    _keywords = []
    if os.path.exists(KEYWORDS_FILE):
        with open(KEYWORDS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith("#"):
                    _keywords.append(line.lower())

    return _keywords


def _check_keywords(question: str) -> bool:
    """Check if question contains any domain keywords."""
    keywords = _load_keywords()
    if not keywords:
        return False

    question_lower = question.lower()
    return any(kw in question_lower for kw in keywords)


# =============================================================================
# FEATURE EXTRACTION (query-only, no retrieval)
# =============================================================================

_WH_WORDS = ["what", "who", "where", "when", "why", "how"]
_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _get_wh_type(question: str) -> str:
    """Determine question type based on first word."""
    words = question.lower().split()
    if not words:
        return "other"

    first = words[0].strip("?.,!")
    if first in _WH_WORDS:
        return first
    return "other"


def _count_entities(question: str) -> int:
    """Count capitalized words (proxy for named entities)."""
    words = question.split()
    count = 0
    for w in words:
        # Skip first word (often capitalized) and short words
        if len(w) > 2 and w[0].isupper() and not w.isupper():
            count += 1
    return count


def extract_features(question: str) -> Dict[str, float]:
    """
    Extract query-only features for routing.

    NO retrieval calls. All features derived from question text.

    Returns:
        Dict of feature name → value
    """
    words = _WORD_PATTERN.findall(question)

    features = {
        "len_words": len(words),
        "len_chars": len(question),
        "has_number": 1.0 if re.search(r"\d", question) else 0.0,
        "num_entities": _count_entities(question),
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0.0,
        "has_question_mark": 1.0 if "?" in question else 0.0,
    }

    # One-hot encode wh-type
    wh = _get_wh_type(question)
    for w in _WH_WORDS + ["other"]:
        features[f"wh_{w}"] = 1.0 if wh == w else 0.0

    return features


def _features_to_vector(features: Dict[str, float]) -> List[float]:
    """Convert feature dict to ordered vector for classifier."""
    # Fixed order for consistency
    keys = [
        "len_words", "len_chars", "has_number", "num_entities",
        "avg_word_length", "has_question_mark",
        "wh_what", "wh_who", "wh_where", "wh_when", "wh_why", "wh_how", "wh_other",
    ]
    return [features.get(k, 0.0) for k in keys]


# =============================================================================
# CLASSIFIER
# =============================================================================

_classifier = None
_classifier_loaded = False


def _load_classifier():
    """Load the trained classifier if available."""
    global _classifier, _classifier_loaded

    if _classifier_loaded:
        return _classifier

    _classifier_loaded = True

    if os.path.exists(ROUTER_MODEL_PATH):
        try:
            _classifier = joblib.load(ROUTER_MODEL_PATH)
            print(f"[router] Loaded classifier from {ROUTER_MODEL_PATH}")
        except Exception as e:
            print(f"[router] Failed to load classifier: {e}")
            _classifier = None
    else:
        print(f"[router] No classifier found at {ROUTER_MODEL_PATH}, using default route")
        _classifier = None

    return _classifier


# =============================================================================
# MAIN ROUTING FUNCTION
# =============================================================================

def route(question: str) -> Route:
    """
    Decide whether to use direct generation or RAG.

    Logic:
    1. If question contains domain keywords → "rag"
    2. If classifier available → use prediction
    3. Otherwise → default route (rag)

    NO retrieval calls. Fast decision based on question text only.
    """
    # Step 1: Keyword check
    if _check_keywords(question):
        return "rag"

    # Step 2: Classifier
    classifier = _load_classifier()

    if classifier is None:
        return DEFAULT_ROUTE

    # Extract features and predict
    features = extract_features(question)
    vector = np.array([_features_to_vector(features)])

    try:
        prediction = classifier.predict(vector)[0]
        return "rag" if prediction == 1 else "direct"
    except Exception as e:
        print(f"[router] Prediction error: {e}")
        return DEFAULT_ROUTE


# =============================================================================
# TRAINING
# =============================================================================

def train(
    questions: List[str],
    labels: List[Route],
    save: bool = True,
) -> Dict:
    """
    Train the router classifier.

    Args:
        questions: List of question strings
        labels: List of "direct" or "rag" labels
        save: Whether to save the model to disk

    Returns:
        Dict with training metrics
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    if len(questions) != len(labels):
        raise ValueError("questions and labels must have same length")

    if len(questions) < 10:
        raise ValueError("Need at least 10 examples to train")

    # Extract features
    print(f"[train] Extracting features from {len(questions)} questions...")
    X = []
    y = []
    for q, label in zip(questions, labels):
        features = extract_features(q)
        X.append(_features_to_vector(features))
        y.append(1 if label == "rag" else 0)

    X = np.array(X)
    y = np.array(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=ROUTER_TRAIN_TEST_SPLIT,
        random_state=ROUTER_RANDOM_SEED,
        stratify=y,
    )

    print(f"[train] Training set: {len(X_train)}, Test set: {len(X_test)}")

    # Train
    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=ROUTER_RANDOM_SEED,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    print(f"[train] Precision: {metrics['precision']:.3f}")
    print(f"[train] Recall: {metrics['recall']:.3f}")
    print(f"[train] F1: {metrics['f1']:.3f}")

    # Save
    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(clf, ROUTER_MODEL_PATH)
        print(f"[train] Model saved to {ROUTER_MODEL_PATH}")

        # Reset loaded classifier so next route() call loads new model
        global _classifier, _classifier_loaded
        _classifier = None
        _classifier_loaded = False

    return metrics


def reload_keywords():
    """Force reload of keywords file."""
    global _keywords
    _keywords = None
    _load_keywords()
