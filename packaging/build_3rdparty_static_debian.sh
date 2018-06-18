#!/bin/sh
BASEDIR=$(dirname $0)
cd $BASEDIR/debian_3rdparty
./build_fftw3.sh
./build_ffmpeg.sh
./build_libsamplerate.sh
./build_zlib.sh
./build_taglib.sh
./build_yaml.sh
./build_chromaprint.sh

if [ "$1" = --with-gaia ];
    then
        ./build_qt.sh
        ./build_gaia.sh
        rm -rf mkspecs plugins translations
fi

rm -rf bin share
