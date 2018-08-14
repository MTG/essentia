#!/bin/sh
. ../build_config.sh
ZLIB_VERSION=zlib-1.2.11

rm -rf tmp
mkdir tmp
cd tmp

echo "Building zlib $ZLIB_VERSION"

curl -SLO https://zlib.net/$ZLIB_VERSION.tar.gz
tar -xf $ZLIB_VERSION.tar.gz
cd $ZLIB_VERSION/

cmake -G "Visual Studio 15 2017 Win64" -DCMAKE_INSTALL_PREFIX=$PREFIX
msbuild.exe install.vcxproj -p:Configuration=Release

mv $PREFIX/bin/zlib.dll $PREFIX/lib

cd ../..
rm -r tmp
