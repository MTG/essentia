#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

wget http://www.fftw.org/$FFTW_VERSION.tar.gz
tar -xf $FFTW_VERSION.tar.gz
cd $FFTW_VERSION

./configure \
    --prefix=$PREFIX \
    $FFTW_FLAGS \
    $SHARED_OR_STATIC
make
make install

cd ../..
rm -r tmp
