#!/bin/sh


PYTHON=python3

# config
MODE=python
CONTEXT=/usr/local/  # run as sudo if the context directoryt is not owned

# setup
$PYTHON setup.py -m $MODE -c $CONTEXT
