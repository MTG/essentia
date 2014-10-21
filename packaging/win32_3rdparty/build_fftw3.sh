#!/bin/sh

FFTW_VERSION=fftw-3.3.2
. ./build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

wget http://www.fftw.org/$FFTW_VERSION.tar.gz
tar -xf $FFTW_VERSION.tar.gz
cd $FFTW_VERSION

./configure \
    --host=$HOST \
    --prefix=$PREFIX \
    --enable-float \
    --with-our-malloc16 \
    --with-windows-f77-mangling \
    --enable-threads \
    --with-combined-threads \
    --enable-portable-binary \
    --enable-sse2 \
    --with-incoming-stack-boundary=2 \
    $SHARED_OR_STATIC
make
make install

cp .libs/libfftw3f-3.dll $PREFIX/lib

cd ../..
rm -r tmp
