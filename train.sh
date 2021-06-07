#!/bin/bash

set -e

PROJECT_DIR=$(eval echo "$(dirname "$0")")
CONFIGS="${PROJECT_DIR}/configs/*"

for FILE in $CONFIGS
do
  # preprocess data
  # train model
  python train.py --config "$FILE"

done

