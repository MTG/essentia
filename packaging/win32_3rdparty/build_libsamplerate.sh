#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

wget http://www.mega-nerd.com/SRC/$LIBSAMPLERATE_VERSION.tar.gz
tar -xf $LIBSAMPLERATE_VERSION.tar.gz
cd $LIBSAMPLERATE_VERSION

./configure \
    --host=$HOST \
    --prefix=$PREFIX \
    --disable-fftw \
    --disable-sndfile \
    $SHARED_OR_STATIC
make
make install

cp src/.libs/libsamplerate-0.dll $PREFIX/lib/libsamplerate.dll

cd ../..
rm -r tmp
