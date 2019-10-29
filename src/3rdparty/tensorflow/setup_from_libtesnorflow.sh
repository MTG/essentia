#!/bin/sh


PYTHON=python3

#configuration
MODE=libtensorflow
PLATFORM=linux
CONTEXT=/usr/local/  # run as sudo if the context directory is not owned
VERSION=1.14.0

# setup
$PYTHON setup.py -m $MODE -p $PLATFORM -c $CONTEXT -v $VERSION
