/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "powerspectrum.h"

using namespace essentia;
using namespace standard;

const char* PowerSpectrum::name = "PowerSpectrum";
const char* PowerSpectrum::description = DOC("This algorithm computes the power spectrum of an array of Reals. The resulting power spectrum is of the same size as the incoming frame.\n"
"\n"
"References:\n"
"  [1] Power Spectrum - from Wolfram MathWorld,\n"
"      http://mathworld.wolfram.com/PowerSpectrum.html");

void PowerSpectrum::configure() {
  // FFT configuration
  _fft->configure("size", parameter("size"));

  // set temp port here as it's not gonna change between consecutive calls
  // to compute()
  _fft->output("fft").set(_fftBuffer);
}

void PowerSpectrum::compute() {

  const std::vector<Real>& signal = _signal.get();
  std::vector<Real>& powerSpectrum = _powerSpectrum.get();

  // no need to make checks regarding the size of the input here, as they
  // will be checked anyway in the FFT algorithm.

  // compute FFT first...
  _fft->input("frame").set(signal);
  _fft->compute();

  // ...and then the square magnitude of it
  powerSpectrum.resize(_fftBuffer.size());
  for (int i=0; i<int(_fftBuffer.size()); ++i) {
    powerSpectrum[i] = _fftBuffer[i].real()*_fftBuffer[i].real() +
                       _fftBuffer[i].imag()*_fftBuffer[i].imag();
  }
}
