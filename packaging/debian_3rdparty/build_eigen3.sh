#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

echo "Building gaia $EIGEN_VERSION"

curl -SLO https://bitbucket.org/eigen/eigen/get/$EIGEN_VERSION.tar.gz
tar -xf $EIGEN_VERSION.tar.gz
cd eigen-eigen*

mkdir build
cd build

cmake ../ -DCMAKE_INSTALL_PREFIX=$PREFIX
make install

cd ../../..
cp share/pkgconfig/eigen3.pc lib/pkgconfig/
rm -rf tmp
