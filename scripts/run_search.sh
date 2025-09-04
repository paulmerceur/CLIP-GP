#!/bin/bash

# Thin wrapper around the Python sweep utility
# Usage: ./scripts/run_search.sh <sweep_yaml> [--devices "0,1" --jobs-per-gpu 1 --retries 0]

if [ $# -lt 1 ]; then
  echo "Usage: $0 <sweep_yaml> [--devices \"0,1\" --jobs-per-gpu 1 --retries 0]"
  exit 1
fi

SWEEP_FILE=$1
shift

python -m utils.hparam_search --sweep-file ${SWEEP_FILE} $@

