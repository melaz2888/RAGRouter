#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/qa
echo "# TODO: download FiQA/HotpotQA subsets and write JSONL with {id,question,answer,domain}" > data/qa/README.txt
echo "Placeholder created. Add your QA files under data/qa/." 
