#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

wget http://www.mega-nerd.com/SRC/$LIBSAMPLERATE_VERSION.tar.gz
tar -xf $LIBSAMPLERATE_VERSION.tar.gz
cd $LIBSAMPLERATE_VERSION

./configure \
    --prefix=$PREFIX \
    $LIBSAMPLERATE_FLAGS \
    $SHARED_OR_STATIC
make
make install

cd ../..
rm -r tmp
