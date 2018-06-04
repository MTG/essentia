#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

echo "Building gaia $GAIA_VERSION"

curl -SLO https://github.com/MTG/gaia/archive/v$GAIA_VERSION.tar.gz
tar -xf v$GAIA_VERSION.tar.gz
cd gaia-$GAIA_VERSION

./waf configure --prefix=$PREFIX
./waf
./waf install 

cd ../..
rm -rf tmp
