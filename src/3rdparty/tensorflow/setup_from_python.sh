#!/bin/sh

cd $(dirname $0)

PYTHON=python3

# config
MODE=python
CONTEXT=/usr/local/  # run as sudo if the context directory is not owned

# setup
$PYTHON setup_tensorflow.py -m $MODE -c $CONTEXT
