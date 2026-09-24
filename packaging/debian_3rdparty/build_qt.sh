#!/usr/bin/env bash
set -e
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

echo "Building qt from $QT_SOURCE_URL"

QT_FILE=${QT_SOURCE_URL##*/}

curl -SLO $QT_SOURCE_URL

tar -xf $QT_FILE

QT_DIR=$(find . -mindepth 1 -maxdepth 1 -type d | head -n 1)
cd "$QT_DIR"

./configure -prefix $PREFIX -shared -opensource -confirm-license $QT_FLAGS

make
make install

cd ../..
rm -fr tmp
