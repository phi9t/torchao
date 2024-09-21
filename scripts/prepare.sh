#!/bin/bash

set -eu -o pipefail

declare -ar REPO_IDS=(
  # meta-llama/Llama-2-7b-chat-hf
  # meta-llama/Meta-Llama-3-8B
  # meta-llama/Meta-Llama-3.1-8B
  # meta-llama/Meta-Llama-3.1-8B-Instruct
  meta-llama/Meta-Llama-3.1-70B-Instruct
  # mistralai/Mistral-7B-Instruct-v0.3
  # mistralai/Mixtral-8x7B-Instruct-v0.1
)

for repo_id in "${REPO_IDS[@]}"; do
  python scripts/download.py --repo_id "${repo_id}"
  python scripts/convert_hf_checkpoint.py --checkpoint_dir "checkpoints/${repo_id}"
  #python scripts/convert_hf_mixtral_moe_checkpoint.py --checkpoint_dir "checkpoints/${repo_id}"
done

# python scripts/download.py --repo_id meta-llama/Llama-2-7b-chat-hf
# python scripts/download.py --repo_id meta-llama/Meta-Llama-3-8B
# python scripts/download.py --repo_id meta-llama/Meta-Llama-3.1-8B
# python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-chat-hf
# python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3-8B
# python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Meta-Llama-3.1-8B
