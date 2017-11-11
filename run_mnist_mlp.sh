#!/bin/bash

SCRIPT_DIR=$(cd $(dirname $0); pwd)
source ~/.pyenv/versions/miniconda3-4.1.11/bin/activate ml
python $SCRIPT_DIR/mnist_mlp.py _input.json

