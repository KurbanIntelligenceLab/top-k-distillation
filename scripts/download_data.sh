#!/bin/bash
# Download the exact dataset files the experiments read (see _CORPUS_CONFIGS in
# code/experiments/common.py) into the Hugging Face cache. The code reads them
# through hf:// paths, so this step is optional: it only avoids downloading at
# run time. Exactness (E1) and Threshold (E2) stream FineWeb-Edu's sample-10BT
# config directly and need no download.
#
# Usage:  bash scripts/download_data.sh
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download
token = os.environ.get("HF_TOKEN") or None
FILES = [
    # (dataset repo, file inside it, used by)
    ("HuggingFaceFW/fineweb-edu", "sample/10BT/000_00000.parquet",
     "general web text: Off-policy, Mechanism, design tests, Capacity, Disagreement (pretrain-style)"),
    ("HuggingFaceH4/ultrachat_200k", "data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
     "On-policy prompts"),
    ("Muennighoff/flan", "train/cnn_dailymail_10templates_train.jsonl",
     "Disagreement (bild-style)"),
    ("glaiveai/glaive-function-calling-v2", "glaive-function-calling-v2.json",
     "Disagreement (toolcall-style)"),
]
for repo, fname, use in FILES:
    path = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", token=token)
    print(f"ok  {repo}/{fname}  ({use})\n    -> {path}")
PY
