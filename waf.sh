#!/usr/bin/env bash

#echo "Deleting old build folder"
#rm -rf build
python_to_use=/usr/bin/python3

CC=clang CXX=clang++ $python_to_use waf configure --build-static  --mode=debug --with-python --include-algos=FileOutputProxy,FrameCutter,MonoLoader,AudioLoader,MonoMixer,Resample,FFT,Magnitude,Windowing,Spectrum,MFCC,MelBands,TriangularBands,DCT,Median,MedianFilter,HPSS,YamlOutput,PoolAggregator --with-example=standard_mfcc,standard_HPSS
$python_to_use waf # -v # --prefix=python_folder

# TODO: use flag to activate install
sudo python3 waf install

# echo "Essentia (hopefully) installed on python3, the address is:"
# echo | which python3

# echo "Running (modified) MFCC example"
# ./build/src/examples/essentia_standard_mfcc 'test/audio/recorded/dubstep.flac' 'output_mfcc_median.yml'
# echo "Running HPSS example"
# ./build/src/examples/essentia_standard_HPSS 'test/audio/recorded/dubstep.flac' 'output_HPSS.yml'

# todo: https://pythonhosted.org/waftools/_modules/waftools/cmake.html waf cmake, to generate cmake from waf