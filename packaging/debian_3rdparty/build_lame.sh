#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

echo "Building lame $LAME_VERSION"

#!/bin/bash
curl -SL -o lame-$LAME_VERSION.tar.gz "http://downloads.sourceforge.net/project/lame/lame/$LAME_VERSION/lame-$LAME_VERSION.tar.gz?r=https%3A%2F%2Fsourceforge.net%2Fprojects%2Flame%2F&ts=1476009914&use_mirror=ufpr"
tar -xf  lame-$LAME_VERSION.tar.gz
cd lame-$LAME_VERSION
CPPFLAGS=-fPIC ./configure --prefix=$PREFIX \
    $SHARED_OR_STATIC
make
make install

cd ../..
rm -r tmp
