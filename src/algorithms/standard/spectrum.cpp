/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
"      http://en.wikipedia.org/wiki/Frequency_spectrum");

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
