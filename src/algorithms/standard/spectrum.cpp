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

#include "spectrum.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Spectrum::name = "Spectrum";
const char* Spectrum::description = DOC("This algorithm calculates the magnitude spectrum of an array of Reals. The resulting magnitude spectrum has a size which is half the size of the input array plus one.\n"
"\n"
"References:\n"
"  [1] Frequency spectrum - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Frequency_spectrum");

void Spectrum::configure() {
  // FFT configuration
  _fft->configure("size", this->parameter("size"));

  // set temp port here as it's not gonna change between consecutive calls
  // to compute()
  _fft->output("fft").set(_fftBuffer);
  _magnitude->input("complex").set(_fftBuffer);
}

void Spectrum::compute() {

  const vector<Real>& signal = _signal.get();
  vector<Real>& spectrum = _spectrum.get();

  // no need to make checks regarding the size of the input here, as they
  // will be checked anyway in the FFT algorithm.

  // compute FFT first...
  _fft->input("frame").set(signal);
  _fft->compute();

  // ...and then the magnitude of it
  _magnitude->output("magnitude").set(spectrum);
  _magnitude->compute();

}
