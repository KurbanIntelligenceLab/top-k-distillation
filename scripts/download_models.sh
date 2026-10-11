#!/bin/bash
# Download every teacher and student model the experiments use (ids from
# code/experiments/common.py) into the Hugging Face cache, so jobs do not
# download at run time. Gated models need an HF_TOKEN (in .env or the
# environment) and an accepted license on the model's Hugging Face page.
#
# Usage:  bash scripts/download_models.sh            # all models (~300 GB)
#         bash scripts/download_models.sh Qwen/Qwen3-0.6B Qwen/Qwen3-8B   # selected ids
#
# Llama-3.1-70B alone is about 140 GB; skip it unless you rerun the 70B Threshold run.
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/.env" ] && { set -a; source "$REPO/.env"; set +a; }

# id                              gated   used as
MODELS=(
  "Qwen/Qwen3-0.6B"               # no    student
  "Qwen/Qwen3-1.7B"               # no    teacher and student
  "Qwen/Qwen3-4B"                 # no    teacher (matched storage)
  "Qwen/Qwen3-8B"                 # no    teacher
  "Qwen/Qwen3-32B"                # no    teacher (Threshold, Systems)
  "meta-llama/Llama-3.2-1B"       # GATED student
  "meta-llama/Llama-3.2-3B"       # GATED teacher (matched storage)
  "meta-llama/Llama-3.1-8B"       # GATED teacher
  "meta-llama/Llama-3.1-70B"      # GATED teacher (Threshold only, ~140 GB)
  "google/gemma-3-1b-pt"          # GATED student
  "google/gemma-3-4b-pt"          # GATED teacher
  "google/gemma-3-12b-pt"         # GATED teacher (Threshold, Systems)
  "HuggingFaceTB/SmolLM2-360M"    # no    student (matched storage)
  "HuggingFaceTB/SmolLM2-1.7B"    # no    teacher (matched storage)
)
# openai/gpt-oss-20b is registered in common.py but its only run was invalid
# (no same-tokenizer student) and is not reported; it is not downloaded here.

if [ "$#" -gt 0 ]; then MODELS=("$@"); fi

for m in "${MODELS[@]}"; do
  echo "=== $m"
  python3 - "$m" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
snapshot_download(repo_id=repo, token=os.environ.get("HF_TOKEN") or None,
                  allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
print("ok", repo)
PY
done
