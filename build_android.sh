#!/usr/bin/env bash
export PATH=$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin:$PATH;
./waf clean;
./waf configure --cross-compile-android --android-target=armv7a --lightweight= --fft=KISS --ignore-algos=LPC --prefix=/Users/carthach/Dev/android/modules/essentia;
./waf;
./waf install;
