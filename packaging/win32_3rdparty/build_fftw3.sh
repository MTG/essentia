#!/usr/bin/env bash
set -e
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

echo "Building fftw $FFTW_VERSION"

curl -SLO http://www.fftw.org/$FFTW_VERSION.tar.gz
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

# TODO Unnecessary?
#cp .libs/libfftw3f-3.dll $PREFIX/lib

cd ../..
rm -r tmp
