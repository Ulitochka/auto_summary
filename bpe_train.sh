#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")

CODE_PATH=$(realpath "${SCRIPT_PATH}"/)
DATA_PATH=$(realpath "${CODE_PATH}"/data/)
MODEL_PATH=$(realpath "${CODE_PATH}"/models/)

PYTHONHASHSEED=128500 PYTHONPATH=${ASPECT_EXTRACTOR_PATH} \
    python3.7 train_subword_model.py \
    --train-path "${DATA_PATH}"/train.csv \
    --model-path "${MODEL_PATH}"/
