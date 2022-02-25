#!/usr/bin/env bash
set -e
BASEDIR=$(dirname $0)
cd $BASEDIR/win32_3rdparty
rm -rf bin dynamic include lib share

./build_eigen3.sh
./build_fftw3.sh
./build_lame.sh
./build_ffmpeg.sh
./build_libsamplerate.sh
./build_zlib.sh
./build_taglib.sh
./build_yaml.sh
./build_chromaprint.sh

rm -rf bin dynamic share
