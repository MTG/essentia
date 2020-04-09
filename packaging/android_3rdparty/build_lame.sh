#!/bin/sh
. ../build_config_android.sh

echo "Building lame $LAME_VERSION"

rm -rf tmp
mkdir tmp
cd tmp

curl -SL -o lame-$LAME_VERSION.tar.gz "http://downloads.sourceforge.net/project/lame/lame/$LAME_VERSION/lame-$LAME_VERSION.tar.gz?r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Flame%2F&ts=1476009914&use_mirror=ufpr"
tar -xf  lame-$LAME_VERSION.tar.gz
cd lame-$LAME_VERSION

CPPFLAGS=-fPIC ./configure --prefix=$PREFIX \
    --disable-fast-install \
    --disable-analyzer-hooks \
    --disable-gtktest \
    --disable-frontend \
    $SHARED_OR_STATIC \
    --host=$ANDROID_TARGET
make
make install

cd ../..
rm -r tmp
