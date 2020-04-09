#!/bin/sh
. ../build_config_android.sh

echo "Building libsamplerate $LIBSAMPLERATE_VERSION"

rm -rf tmp
mkdir tmp
cd tmp

curl -SLO http://www.mega-nerd.com/SRC/$LIBSAMPLERATE_VERSION.tar.gz
tar -xf $LIBSAMPLERATE_VERSION.tar.gz

# Libsamplerate contains outdated config.sub and config.guess.
# Replace them with the newest version to make Android builds work.
git clone --depth=1 git://git.savannah.gnu.org/config.git
mv config/config.sub config/config.guess $LIBSAMPLERATE_VERSION/Cfg

cd $LIBSAMPLERATE_VERSION

CPPFLAGS=-fPIC ./configure \
    --prefix=$PREFIX \
    $LIBSAMPLERATE_FLAGS \
    $SHARED_OR_STATIC \
    --host=$ANDROID_TARGET
make
make install


cd ../..
rm -rf tmp
