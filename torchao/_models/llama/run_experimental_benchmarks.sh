#!/bin/bash

set -eu -o pipefail

# This script runs the benchmarks for the Meta-LLAMA models.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
readonly SCRIPT_DIR

export CHECKPOINT_PATH="${SCRIPT_DIR}/../../../checkpoints" # path to checkpoints folder
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Checkpoints folder not found. Please set the correct path to the checkpoints folder."
    exit 1
fi

# https://github.com/pytorch-labs/gpt-fast
export MODEL_REPO=meta-llama/Meta-Llama-3.1-8B

export MODEL_PATH="${CHECKPOINT_PATH}/${MODEL_REPO}/model.pth"
if [ ! -f "${MODEL_PATH}" ]; then
    echo "Model checkpoint not found. Please set the correct path to the model checkpoint."
    exit 1
fi

python experimental_benchmark.py --checkpoint_path "${MODEL_PATH}" --compile --write_result benchmark_results.txt
python experimental_benchmark.py --checkpoint_path "${MODEL_PATH}" --compile --quantization int8dq --write_result benchmark_results.txt
python experimental_benchmark.py --checkpoint_path "${MODEL_PATH}" --compile --quantization int8wo --write_result benchmark_results.txt
python experimental_benchmark.py --checkpoint_path "${MODEL_PATH}" --compile --quantization int4wo-64 --write_result benchmark_results.txt

# NOTE: don't worry about this yet...
# python experimental_benchmark.py --checkpoint_path "${MODEL_PATH}" --compile --compile_prefill --quantization autoquant-int4 --write_result benchmark_results.txt
# python experimental_benchmark.py --checkpoint_path "${MODEL_PATH}" --compile --quantization fp6 --write_result benchmark_results.txt --precision float16
