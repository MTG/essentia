#!/bin/sh

cd $(dirname $0)

PYTHON=python3

#configuration
MODE=libtensorflow
PLATFORM=linux
CONTEXT=/usr/local/  # run as sudo if the context directory is not owned
VERSION=1.15.0

# setup
$PYTHON setup_tensorflow.py -m $MODE -p $PLATFORM -c $CONTEXT -v $VERSION
