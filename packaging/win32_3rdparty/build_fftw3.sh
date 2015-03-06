#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

wget http://www.fftw.org/$FFTW_VERSION.tar.gz
tar -xf $FFTW_VERSION.tar.gz
cd $FFTW_VERSION

./configure \
    --host=$HOST \
    --prefix=$PREFIX \
    $FFTW_FLAGS \
    --with-windows-f77-mangling \
    --enable-threads \
    --with-combined-threads \
    --enable-portable-binary \
    $SHARED_OR_STATIC
make
make install

cp .libs/libfftw3f-3.dll $PREFIX/lib

cd ../..
rm -r tmp
