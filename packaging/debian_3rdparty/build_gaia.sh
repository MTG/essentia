#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

echo $PREFIX

wget https://github.com/MTG/gaia/archive/v$GAIA_VERSION.tar.gz -O gaia-$GAIA_VERSION.tar.gz
tar -xf gaia-$GAIA_VERSION.tar.gz
cd gaia-$GAIA_VERSION

./waf configure --prefix=$PREFIX
./waf
./waf install 

cd ../..
rm -rf tmp
