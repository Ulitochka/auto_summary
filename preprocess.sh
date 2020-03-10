#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")

CODE_PATH=$(realpath "${SCRIPT_PATH}"/)
DATA_PATH=$(realpath "${CODE_PATH}"/data/)
VOCAB_PATH=$(realpath "${CODE_PATH}"/vocs/)
CONFIG_PATH=$(realpath "${CODE_PATH}"/configs/)

PYTHONHASHSEED=128500 PYTHONPATH=${ASPECT_EXTRACTOR_PATH} \
    python3.7 preprocess.py \
    --train-path "${DATA_PATH}"/train.csv \
    --vocabulary-path "${VOCAB_PATH}"/ \
    --config-path "${CONFIG_PATH}"/custom.json
