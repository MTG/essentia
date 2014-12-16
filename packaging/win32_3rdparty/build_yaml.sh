#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

echo $PREFIX


wget http://pyyaml.org/download/libyaml/$LIBYAML_VERSION.tar.gz
tar -xf $LIBYAML_VERSION.tar.gz
cd $LIBYAML_VERSION

# fails to compile shared library, building only static
CFLAGS="-DYAML_DECLARE_STATIC" ./configure \
    --host=$HOST \
    --prefix=$PREFIX \
    $SHARED_OR_STATIC
make
make install

cd ../..
rm -r tmp
