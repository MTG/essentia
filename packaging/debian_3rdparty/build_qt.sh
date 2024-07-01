#!/usr/bin/env bash
set -e
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

echo "Building qt from $QT_SOURCE_URL"

QT_SHORT_VERSION=${QT_VERSION%.*}
QT_SOURCE_URL=https://download.qt.io/archive/qt/$QT_SHORT_VERSION/$QT_VERSION/single/qt-everywhere-opensource-src-$QT_VERSION.tar.xz
QT_FILE=${QT_SOURCE_URL##*/}
QT_FILE_DIR=qt-everywhere-src-$QT_VERSION

curl -SLO $QT_SOURCE_URL
tar -xf $QT_FILE
cd $QT_FILE_DIR

./configure -prefix $PREFIX -static -opensource -confirm-license $QT_FLAGS

make
make install

cd ../..
rm -fr tmp
