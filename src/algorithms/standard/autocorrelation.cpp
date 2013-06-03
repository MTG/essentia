/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "autocorrelation.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* AutoCorrelation::name = "AutoCorrelation";
const char* AutoCorrelation::description = DOC(
"This algorithm returns the autocorrelation vector of a signal.\n"
"It uses the version most commonly used in signal processing, which doesn't remove "
"the mean from the observations.\n"
"\n"
"References:\n"
"  [1] Autocorrelation -- from Wolfram MathWorld,\n"
"      http://mathworld.wolfram.com/Autocorrelation.html\n"
"  [2] Autocorrelation - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Autocorrelation");


void AutoCorrelation::configure() {
  string ntype = parameter("normalization").toString();

  if (ntype == "standard") {
    _unbiasedNormalization = false;
  }
  else if (ntype == "unbiased") {
    _unbiasedNormalization = true;
  }

  _fft->output("fft").set(_fftBuffer);
  _ifft->input("fft").set(_fftBuffer);
}

void AutoCorrelation::compute() {

  const std::vector<Real>& signal = _signal.get();
  vector<Real>& correlation = _correlation.get();

  if (signal.size() == 0) {
    correlation.resize(0);
    return;
  }

  _fft->input("frame").set(_paddedSignal);
  _ifft->output("frame").set(_corr);

  int size = int(signal.size());
  int sizeFFT = int(nextPowerTwo(2*size));

  // formula to get the auto-correlation (in matlab) is:
  //  [M,N] = size(x)
  //  X = fft(x,2^nextpow2(2*M-1));
  //  c = ifft(abs(X).^2);

  // copy signal, and zero-pad it; use _corr as temporary array
  _paddedSignal.resize(sizeFFT);
  for (int i=0; i<size; i++) _paddedSignal[i] = signal[i];
  for (int i=size; i<sizeFFT; i++) _paddedSignal[i] = 0.0;

  // first compute fft
  _fft->compute();

  // take squared amplitude of the spectrum
  // (using magnitude would compute sqrt*sqrt)
  for (int i=0; i<int(_fftBuffer.size()); i++) {
    _fftBuffer[i] = complex<Real>(_fftBuffer[i].real() * _fftBuffer[i].real() +
                                  _fftBuffer[i].imag() * _fftBuffer[i].imag(),
                                  0.0); // squared amplitude -> complex part = 0
  }

  // step 3
  _ifft->compute();

  // copy results in output array, scaling on the go (normalizing the output of the IFFT)
  Real scale = 1.0 / sizeFFT;
  correlation.resize(size);

  if (_unbiasedNormalization) {
    for (int i=0; i<size; i++) {
      correlation[i] = _corr[i] * scale / (size - i);
    }
  }
  else {
    for (int i=0; i<size; i++) {
      correlation[i] = _corr[i] * scale;
    }
  }

}
