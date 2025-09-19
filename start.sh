#!/bin/bash

# import secrets
test -f $GET_SECRETS_FILE && source $GET_SECRETS_FILE

export PROMETHEUS_MULTIPROC_DIR=${LLM_METRICS_PATH:-/tmp/metrics}; 
mkdir -p $PROMETHEUS_MULTIPROC_DIR;

if [[ -f ./dist/entrypoint/entrypoint ]]; then 
    ./dist/entrypoint/entrypoint   
else 
    python3 entrypoint.py;
fi
