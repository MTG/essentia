/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of Essentia
 *
 * Essentia is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 */

#include "powerspectrum.h"

using namespace essentia;
using namespace standard;

const char* PowerSpectrum::name = "PowerSpectrum";
const char* PowerSpectrum::description = DOC("This algorithm computes the power spectrum of an array of Reals. The resulting power spectrum is of the same size as the incoming frame.\n"
"\n"
"References:\n"
"  [1] Power Spectrum - from Wolfram MathWorld,\n"
"  http://mathworld.wolfram.com/PowerSpectrum.html");

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
