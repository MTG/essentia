#!/bin/sh
. ../build_config_android.sh

echo "Building libyaml $LIBYAML_VERSION"

rm -rf tmp
mkdir tmp
cd tmp

curl -SLO http://pyyaml.org/download/libyaml/$LIBYAML_VERSION.tar.gz
tar -xf $LIBYAML_VERSION.tar.gz

cd $LIBYAML_VERSION
CFLAGS="-fPIC" ./configure \
    --prefix=$PREFIX \
    $SHARED_OR_STATIC \
    --host=$ANDROID_TARGET \
    --build=$ANDROID_BUILD_ARCH
make
make install

cd ../..
rm -r tmp
