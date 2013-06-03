#!/bin/sh

ALGOINCLUDE=Spectrum,MonoLoader,AudioLoader,MonoMixer,Resample,TonalExtractor,FrameCutter,Windowing,Spectrum,SpectralPeaks,HPCP,Key,YamlOutput,ChordsDescriptors,ChordsDetection,PeakDetection,NoiseAdder,FFTW,Magnitude,LowPass,IIR
ALGOINCLUDE_OPT="algoinclude=$ALGOINCLUDE"

scons $ALGOINCLUDE_OPT
scons $ALGOINCLUDE_OPT examples
