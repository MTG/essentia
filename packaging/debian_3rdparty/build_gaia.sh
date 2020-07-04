#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

echo "Building gaia $GAIA_VERSION"

PKG_CONFIG_PATH=$(pkg-config --variable pc_path pkg-config)
PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH

curl -SLO https://github.com/MTG/gaia/archive/v$GAIA_VERSION.tar.gz
tar -xf v$GAIA_VERSION.tar.gz
cd gaia-$GAIA_VERSION

./waf configure --prefix=$PREFIX --pkg-config-path=PKG_CONFIG_PATH
./waf
./waf install 

cd ../..
rm -rf tmp
