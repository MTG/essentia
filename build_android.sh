#!/usr/bin/env bash
export PATH=~/Dev/android/toolchain/bin:$PATH;
./waf clean;
./waf configure --cross-compile-android --lightweight= --fft=KISS --ignore-algos=LPC --prefix=/Users/carthach/Dev/android/modules/essentia;
./waf;
./waf install;
