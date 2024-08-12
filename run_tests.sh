#!/bin/bash

CONFIG_FOLDER=""

python3 main.py \
    --config="configs/weighted.yaml"

python3 main.py \
    --config="configs/unweighted.yaml"