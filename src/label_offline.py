# TODO: implement heuristics + small manual loop.
# For Day-1 skeleton, we simply create an empty labels file if absent.
import os, json, sys
from pathlib import Path

OUT = Path("data/labels.jsonl")

def main():
    if OUT.exists():
        print("labels.jsonl already exists.")
        return
    with open(OUT, "w") as f:
        pass
    print("Created empty data/labels.jsonl. Fill it with {id, question, route}.")

if __name__ == "__main__":
    main()
