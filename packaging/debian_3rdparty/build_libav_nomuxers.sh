#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

wget https://libav.org/releases/$LIBAV_VERSION.tar.gz
tar xf $LIBAV_VERSION.tar.gz
cd $LIBAV_VERSION

./configure \
    $LIBAV_AUDIO_FLAGS \
    --prefix=$PREFIX \
    $SHARED_OR_STATIC
make
make install

cd ../..
rm -r tmp
