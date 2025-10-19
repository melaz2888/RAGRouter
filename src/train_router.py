import json, os
from pathlib import Path

def main():
    labels_path = Path("data/labels.jsonl")
    if not labels_path.exists() or labels_path.stat().st_size == 0:
        print("No labels found. Run src/label_offline.py and populate data/labels.jsonl first.")
        return
    # TODO: load features for each question, train LR, save models/router.joblib and models/threshold.txt
    print("Training placeholder: implement feature extraction + LR fit here.")

if __name__ == "__main__":
    main()
