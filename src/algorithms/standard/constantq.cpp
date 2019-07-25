/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#include "constantq.h"
#include "essentia.h"
#include <iostream>

using namespace std;
using namespace essentia;
using namespace standard;

const char* ConstantQ::name = "ConstantQ";
const char* ConstantQ::category = "Standard";
const char* ConstantQ::description = DOC("This algorithm computes Constant Q Transform given the FFT for fast calculation. It transforms a standard FFT into the log frequency domain.\n"
"\n"
"References:\n"
"  [1] Constant Q transform - Wikipedia, the free encyclopedia,\n"
"  https://en.wikipedia.org/wiki/Constant_Q_transform\n"
"  [2] Brown, J. C., & Puckette, M. S. (1992). An efficient algorithm for the\n"
"  calculation of a constant Q transform. The Journal of the Acoustical Society\n"
"  of America, 92(5), 2698-2701.\n"
"  [3] Schörkhuber, C., & Klapuri, A. (2010). Constant-Q transform toolbox for\n"
"  music processing. In 7th Sound and Music Computing Conference, Barcelona,\n"
"  Spain (pp. 3-64).");


void ConstantQ::compute() {

  const vector<complex<Real> >& fft = _fft.get();
  vector<complex<Real> >& constantQ = _constantQ.get();

  if (fft.size() != _inputFFTSize) {
    throw EssentiaException("ConstantQ: input FFT size must be equal to: ", _inputFFTSize);
  }

  constantQ.assign(_numberBins, 0.0 + 0.0j); // Initialize output.

  const struct SparseKernel &sk = _sparseKernel;
  for (unsigned i=0; i<sk.real.size(); i++) {
    const unsigned row = sk.j[i];
    const unsigned col = sk.i[i];
    const double & r1  = sk.real[i];
    const double & i1  = sk.imag[i];
    const double r2 = (double) fft[col].real();
    const double i2 = (double) fft[col].imag();
    constantQ[row] += complex<Real>((r1*r2 - i1*i2), (r1*i2 + i1*r2));
  }
}


void ConstantQ::configure() {
  _sampleRate = parameter("sampleRate").toDouble();
  _minFrequency = parameter("minFrequency").toDouble();
  _numberBins = parameter("numberBins").toInt();
  _binsPerOctave = parameter("binsPerOctave").toInt();
  _threshold = parameter("threshold").toDouble();

  // Constant Q factor (resolution of filter windows, larger values correspond
  // to longer windows.
  // TODO make filterScale configurable (0,+inf) (default = 1)?
  const Real filterScale = 1.;
  _Q = filterScale / (pow(2, (1/(double)_binsPerOctave))-1);
  
  // The largest window size we'll require. We center-pad filters to the next
  // power of two of the maximum filter length.
  _windowSize = nextPowerTwo((int) ceil(_Q *_sampleRate/_minFrequency));

  // Work only with a non-negative part of FFT as an input
  _inputFFTSize = _windowSize/2+1;

  struct SparseKernel &sk = _sparseKernel;
  sk.i.reserve(_windowSize);
  sk.j.reserve(_windowSize);
  sk.real.reserve(_windowSize);
  sk.imag.reserve(_windowSize);
  
  vector<complex<Real> > binKernel;
  vector<complex<Real> > binKernelFFT;

  // For each bin value k, calculate temporal kernel, take its FFT to
  // calculate the spectral kernel then threshold it to make it sparse and
  // add it to the sparse kernels matrix.
  
  for (unsigned k=_numberBins; k>0; k--) {

    const double frequency = _minFrequency * pow(2,((double)(k))/(double)_binsPerOctave);
    const unsigned length = (int) ceil(_Q * _sampleRate / frequency);

    // TODO throw an exception if the filter pass-band lies beyond Nyquist
    // similar to https://github.com/librosa/librosa/blob/master/librosa/filters.py

    // We can get the window function from the output of Windowing algorithm
    // with the unity input.
    // TODO configure windowing: zeroPhase=False; try zeroPadding
    // TODO make window type configurable
    vector<Real> unity(length, 1.);
    vector<Real> window;
    _windowing->input("frame").set(unity);
    _windowing->output("frame").set(window);
    _windowing->compute();

    // Compute temporal kernel
    binKernel.assign(_windowSize, 0.0 + 0.0j);
    unsigned origin = _windowSize/2 - length/2;
    for (int i=0; i<length; i++) {
      const double angle = 2 * M_PI * _Q * i / length;
      binKernel[origin + i] = complex <Real>(window[i]*cos(angle), window[i]*sin(angle));
    }

    // Compute FFT of temporal kernel.
    _fftc->input("frame").set(binKernel);
    _fftc->output("fft").set(binKernelFFT);
    _fftc->compute();

    for (int j=0; j<binKernelFFT.size(); j++) {
      // Perform thresholding to make the kernel sparse: keep values with
      // absolute value above the threshold.
      if (abs(binKernelFFT[j]) <= _threshold) continue;

      // Insert non-zero position indexes.
      sk.i.push_back(j);
      sk.j.push_back(k);

      // Take conjugate, normalize and add to array sparseKernel.
      sk.real.push_back(((complex<double>) binKernelFFT[j]).real()/_windowSize);
      sk.imag.push_back(-((complex<double>) binKernelFFT[j]).imag()/_windowSize);
    }
  }
}
