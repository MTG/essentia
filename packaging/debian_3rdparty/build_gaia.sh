#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

echo $PREFIX

git clone https://github.com/MTG/gaia.git
cd gaia
git checkout $GAIA_VERSION

./waf configure --prefix=$PREFIX
./waf
./waf install 

cd ../..
rm -rf tmp
