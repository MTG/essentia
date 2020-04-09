#!/bin/sh
. ../build_config_android.sh

rm -rf tmp
mkdir tmp
cd tmp

echo "Building zlib $ZLIB_VERSION"

curl -SLO https://zlib.net/$ZLIB_VERSION.tar.gz
tar -xf $ZLIB_VERSION.tar.gz
cd $ZLIB_VERSION/


CC=$CC CFLAGS=-fPIC ./configure \
    --prefix=$PREFIX
make
make install

cd ../..
rm -r tmp

