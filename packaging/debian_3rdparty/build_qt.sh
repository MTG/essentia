#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

echo $PREFIX

QT_FILE=${QT_SOURCE_URL##*/}

wget $QT_SOURCE_URL

tar -xf $QT_FILE
cd $(basename $QT_FILE .tar.gz)

./configure -prefix $PREFIX -static -opensource -confirm-license $QT_FLAGS

make
make install

cd ../..
rm -fr tmp
