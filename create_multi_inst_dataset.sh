#!/bin/bash

set -e

PROJECT_DIR=$(eval echo "$(dirname "$0")")
CONFIGS="${PROJECT_DIR}/configs/*"
INSTRUMENTS="${PROJECT_DIR}/instruments.txt"

python create_datasets.py --instruments "${INSTRUMENTS}"


