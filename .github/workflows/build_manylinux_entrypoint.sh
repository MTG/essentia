#!/bin/bash
set -e -x

ln -s /github/workspace /io
cd /io
source travis/build_wheels.sh
ls
