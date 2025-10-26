#!/bin/bash

ROLL_PATH="/workspace/ROLL-main"
CONFIG_PATH=$(basename $(dirname $0))
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"
python examples/start_agentic_pipeline.py \
    --config_path $CONFIG_PATH \
    --config_name agentic_click_and_read

