#!/bin/sh
. ../build_config.sh

echo "Building FFmpeg $FFMPEG_VERSION"

mux=$1
if test "$1" = "--no-muxers"; then
    echo Building FFmpeg without muxers
    FFMPEG_AUDIO_FLAGS_MUXERS=""
fi

rm -rf tmp
mkdir tmp
cd tmp

curl -SLO https://ffmpeg.org/releases/$FFMPEG_VERSION.tar.gz
tar xf $FFMPEG_VERSION.tar.gz
cd $FFMPEG_VERSION

./configure \
    --enable-pic \
    $FFMPEG_AUDIO_FLAGS \
    $FFMPEG_AUDIO_FLAGS_MUXERS \
    --prefix=$PREFIX \
    $SHARED_OR_STATIC
make
make install

cd ../..
rm -r tmp
