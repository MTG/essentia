#!/bin/sh
. ../build_config.sh

mux=$1
if test "$1" = "--no-muxers"; then
    echo Building FFmpeg without muxers
    FFMPEG_AUDIO_FLAGS_MUXERS=""
fi

rm -rf tmp
mkdir tmp
cd tmp

wget https://ffmpeg.org/releases/$FFMPEG_VERSION.tar.gz
echo DEBUG - Unpacking
tar xf $FFMPEG_VERSION.tar.gz
cd $FFMPEG_VERSION

echo DEBUG - configuring
./configure \
    --toolchain=msvc \
    $FFMPEG_AUDIO_FLAGS \
    $FFMPEG_AUDIO_FLAGS_MUXERS \
    --prefix=$PREFIX \
    --arch=x86_32 \
    --enable-memalign-hack \
    --enable-shared \
    --disable-static

    #--enable-cross-compile \
    #--cross-prefix=$HOST- \
    #--target-os=mingw32 \
    #--enable-w32threads \
    #$SHARED_OR_STATIC

echo DEBUG - making
make
make install

cp libavutil/*.dll $PREFIX/lib
cp libavcodec/*.dll $PREFIX/lib
cp libavformat/*.dll $PREFIX/lib
cp libavresample/*.dll $PREFIX/lib

cd ../..
rm -r tmp
