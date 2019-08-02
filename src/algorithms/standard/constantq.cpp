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
const char* ConstantQ::description = DOC("This algorithm computes Constant Q Transform using the FFT for fast calculation. It transforms a windowed audio frame into the log frequency domain.\n"
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

  const vector<Real> & frame = _frame.get();
  vector<complex<Real> >& constantQ = _constantQ.get();

  if (frame.size() != _windowSize) {
    throw EssentiaException("ConstantQ: input frame size must be equal to: ", _windowSize);
  }

  _fft->input("frame").set(frame);
  _fft->compute();

  constantQ.assign(_numberBins, complex<Real>(0, 0)); // Initialize output.

  for (unsigned i=0; i<_sparseKernel.real.size(); i++) {
    const unsigned row = _sparseKernel.j[i];
    const unsigned col = _sparseKernel.i[i];
    const double & r1  = _sparseKernel.real[i];
    const double & i1  = _sparseKernel.imag[i];
    const double r2 = (double)_fftData[col].real();
    const double i2 = (double)_fftData[col].imag();

    constantQ[row] += complex<Real>((r1*r2 - i1*i2), (r1*i2 + i1*r2));
  }
}


void ConstantQ::configure() {
  _sampleRate = parameter("sampleRate").toDouble();
  _minFrequency = parameter("minFrequency").toDouble();
  _numberBins = parameter("numberBins").toInt();
  _binsPerOctave = parameter("binsPerOctave").toInt();
  _threshold = parameter("threshold").toDouble();
  _scale = parameter("scale").toDouble();
  _minimumKernelSize = parameter("minimumKernelSize").toInt();
  _zeroPhase = parameter("zeroPhase").toBool();

  _windowing->configure("type", parameter("windowType").toString());

  // Constant Q factor (resolution of filter windows, larger values correspond
  // to longer windows.
  _Q = _scale / (pow(2, (1/(double)_binsPerOctave))-1);
  
  // The largest window size we'll require. We center-pad filters to the next
  // power of two of the maximum filter length.
  _windowSize = nextPowerTwo((int)ceil(_Q * _sampleRate / _minFrequency));

  _fft->configure("size", _windowSize);
  _fft->output("fft").set(_fftData);

  // Work only with a non-negative part of FFT as an input.
  _inputFFTSize = _windowSize / 2 + 1;

  // Get a new sparseKernel and reserve the maximum amount
  // of memory posible (for the dense kernel case).
  _sparseKernel = {};
  _sparseKernel.i.reserve(_windowSize / 2 + 1);
  _sparseKernel.j.reserve(_windowSize / 2 + 1);
  _sparseKernel.real.reserve(_windowSize / 2 + 1);
  _sparseKernel.imag.reserve(_windowSize / 2 + 1);

  vector<complex<Real> > binKernel;
  vector<complex<Real> > binKernelFFT;

  // For each bin value k, calculate temporal kernel, take its FFT to
  // calculate the spectral kernel then threshold it to make it sparse and
  // add it to the sparse kernels matrix.
  for (int k = _numberBins - 1; k > -1; k--) {

    const double frequency = _minFrequency * pow(2,((double)(k)) / (double)_binsPerOctave);
    Real length = _Q * _sampleRate / frequency;

    // We can get the window function from the output of Windowing algorithm
    // with the unity input.
    unsigned int ilen = max((2 * (floor(length) / 2)), (Real)_minimumKernelSize);

    vector<Real> unity(ilen, 1.);
    vector<Real> window;
    _windowing->input("frame").set(unity);
    _windowing->output("frame").set(window);
    _windowing->compute();

    // Make sure that every filter is below the Nyquist frequency.
    // It is enough if we check the higher channel.
    if (k == (int)_numberBins - 1) {
      Real windowBW = length * sumSquare(window) / pow(sum(window), 2);
      Real passBand = frequency * (1 + 0.5 * windowBW / _Q);
      if (passBand > _sampleRate / 2){
        ostringstream msg;
        msg << "ConstantQ: Attempted to create a filter whose pass-band (" << passBand
            << " Hz) is beyond the Nyquist frequency (" << _sampleRate / 2
            << " Hz). A possible fix is to reduce the 'numberBins' parameter";
        throw EssentiaException(msg.str());
      }
    }

    // Compute the threshold directly from the window.
    vector<Real> sortedWindow(window);
    sort(sortedWindow.begin(), sortedWindow.end());
    
    // Notice that Essentia's windows are scaled by 2 so
    // first we have to renormalize.
    normalizeSum(sortedWindow);
    
    // If we do not manage to compute a threshold the default value
    // is high so the sparse matrix will be empty to make us notice
    // that something went wrong.
    Real threshold = 1;
    Real cumSum = 0;
  
    // Look for the index of the desired quantile.
    for (size_t i = 0; i <sortedWindow.size(); i++) {
      cumSum += sortedWindow[i];

      if (cumSum > _threshold) {
        threshold = sortedWindow[i];
        break;
      }
    }

    // Compute temporal kernels
    binKernel.assign(_windowSize, complex<Real>(0, 0));
    unsigned origin = _zeroPhase ? _windowSize - (int)ilen / 2 : _windowSize / 2 - (int)ilen / 2;
    
    Real a = -(Real)ilen / 2.0;
    for (unsigned i = 0; i < ilen; i++, a++) {
      const double angle = 2.0 * M_PI * a * frequency / _sampleRate;
      const unsigned int idx = _zeroPhase ? (origin + i) % _windowSize : origin + i;
      binKernel[idx] = window[i] * complex<Real>(cos(angle), sin(angle));
    }

    // Compute FFT of temporal kernel.
    _fftc->input("frame").set(binKernel);
    _fftc->output("fft").set(binKernelFFT);
    _fftc->compute();

    for (size_t j=0; j<binKernelFFT.size(); j++) {
      // Perform thresholding to make the kernel sparse: keep values with
      // absolute value above the threshold.
      if (abs(binKernelFFT[j]) / 2 <= threshold) continue;

      // Insert non-zero position indexes.
      _sparseKernel.i.push_back(j);
      _sparseKernel.j.push_back(k);

      // Take conjugate, normalize and add to array sparseKernel.
      _sparseKernel.real.push_back(binKernelFFT[j].real() * length / ((Real)_windowSize * 2));
      _sparseKernel.imag.push_back(-binKernelFFT[j].imag() * length / ((Real)_windowSize * 2));
    }
  }
}
