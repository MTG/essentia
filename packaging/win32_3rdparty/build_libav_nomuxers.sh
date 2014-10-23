#!/bin/sh
. ../build_config.sh

rm -rf tmp
mkdir tmp
cd tmp

wget https://libav.org/releases/$LIBAV_VERSION.tar.gz
tar xf $LIBAV_VERSION.tar.gz
cd $LIBAV_VERSION

./configure \
    $LIBAV_AUDIO_FLAGS \
    --prefix=$PREFIX \
    --enable-cross-compile \
    --cross-prefix=$HOST- \
    --arch=x86_32 \
    --target-os=mingw32 \
    --enable-memalign-hack \
    #--enable-w32threads \
    $SHARED_OR_STATIC
make
make install

cp libavutil/avutil.dll $PREFIX/lib
cp libavutil/avutil-51.dll $PREFIX/lib
cp libavcodec/avcodec.dll $PREFIX/lib
cp libavcodec/avcodec-53.dll $PREFIX/lib
cp libavformat/avformat.dll $PREFIX/lib
cp libavformat/avformat-53.dll $PREFIX/lib

cd ../..
rm -r tmp
