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
tar xf $FFMPEG_VERSION.tar.gz
cd $FFMPEG_VERSION

./configure \
    $FFMPEG_AUDIO_FLAGS \
    $FFMPEG_AUDIO_FLAGS_MUXERS \
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
