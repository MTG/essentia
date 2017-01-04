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
#include "essentiamath.h"
#include <iostream>

using namespace std;
using namespace essentia;
using namespace standard;

const char* ConstantQ::name = "ConstantQ";
const char* ConstantQ::category = "Standard";
const char* ConstantQ::description = DOC("This algorithm implements Constant Q Transform using the FFT for fast calculation.\n"
"\n"
"References:\n"
"  [1] Constant Q transform - Wikipedia, the free encyclopedia,\n"
"  https://en.wikipedia.org/wiki/Constant_Q_transform");
// TODO: add references

static double nextpow2(double x) {
  // returns the smallest integer n such that 2^n >= x.
  return ceil(log(x)/log(2.0));
}

static double squaredModule(const complex<Real> xx) {
  complex<double> multComplex = (complex<double>) xx * (complex<double>) xx;
  return multComplex.real() + multComplex.imag();
}


void ConstantQ::compute() {

  const vector<complex<Real> >& fft = _fft.get();
  vector<complex<Real> >& constantQ = _constantQ.get();

  if (!_sparseKernel) {
    throw EssentiaException("ERROR: ConstantQ::compute: Sparse kernel has not been initialized");
  }

  if (fft.size() != _FFTLength) {
    throw EssentiaException("ERROR: ConstantQ::compute: input FFT size must be equal to: ", _FFTLength);
  }

  SparseKernel *sk = _sparseKernel;

  constantQ.assign(_numberBins, 0.0 + 0.0j); // initialize output

  const unsigned int sparseCells = sk->real.size();

  for (unsigned i=0; i<sparseCells; i++) {
    const unsigned row = sk->j[i];
    const unsigned col = sk->i[i];
    const double & r1  = sk->real[i];
    const double & i1  = sk->imag[i];
    const double & r2  = (double) fft[_FFTLength - col - 1].real();
    const double & i2  = (double) fft[_FFTLength - col - 1].imag();
    // add the multiplication
    constantQ[row] += complex <Real>((r1*r2 - i1*i2), (r1*i2 + i1*r2));
  }    
}


void ConstantQ::configure() {
  _sampleRate = parameter("sampleRate").toDouble();
  _minFrequency = parameter("minFrequency").toDouble();
  _numberBins = parameter("numberBins").toInt();
  _binsPerOctave = parameter("binsPerOctave").toInt();
  _threshold = parameter("threshold").toDouble();

  // Constant Q factor
  // TODO make filterScale configurable (0,+inf) (default = 1)
  // Resolution of filter windows (larger values correspond to longer windows).
  const Real filterScale = 1.;
  _Q = filterScale / (pow(2, (1/(double)_binsPerOctave))-1);
  
  // We'll center-pad filters to the next power of two of the maximum filter length
  _FFTLength = (int) pow(2, nextpow2(ceil(_Q *_sampleRate/_minFrequency)));

  SparseKernel *sk = new SparseKernel();

  // Initialize temporal kernel with zeros
  vector<complex<Real> > hammingWindow(_FFTLength, 0.0 + 0.0j);
  vector<complex<Real> > transfHammingWindow(_FFTLength, 0.0 + 0.0j);

  // Twice length to deal with complex numbers
  sk->i.reserve( _FFTLength*2 );
  sk->j.reserve( _FFTLength*2 );
  sk->real.reserve( _FFTLength*2 );
  sk->imag.reserve( _FFTLength*2 );
  
  // For each bin value k, calculate temporal kernel, take its fft to
  // calculate the spectral kernel then threshold it to make it sparse and
  // add it to the sparse kernels matrix
  double squareThreshold = _threshold * _threshold;

  for (unsigned k=_numberBins; k--; ) {

    // Compute a hamming window
    hammingWindow.assign(_FFTLength, 0.0 + 0.0j);
    const double binFrequency = _minFrequency * pow(2,((double)(k))/(double)_binsPerOctave)
    // TODO check that the filter pass-band does not lie beyond Nyquist
    // similar to https://github.com/librosa/librosa/blob/master/librosa/filters.py#L625


    const unsigned hammingLength = (int) ceil(_Q * _sampleRate / binFrequency);
    unsigned origin = _FFTLength/2 - hammingLength/2;

    for (int i=0; i<hammingLength; i++) {
      const double angle = 2 * M_PI * _Q * i/hammingLength;
      const double absol = hamming(hammingLength, i)/hammingLength;
      hammingWindow[origin + i] = complex <Real>(absol*cos(angle), absol*sin(angle));
    }

    // TODO why?
    for (int i=0; i <_FFTLength/2; ++i) {
      complex<Real> temp = hammingWindow[i];
      hammingWindow[i] = hammingWindow[i + _FFTLength/2];
      hammingWindow[i + _FFTLength/2] = temp;
    }

    // Compute FFT of the hamming window
    _fftc->input("frame").set(hammingWindow);
    _fftc->output("fft").set(transfHammingWindow);
    _fftc->compute();

    // Increase the output size of the FFT to _FFTLength by mirroring the data
    int ind = transfHammingWindow.size() - 1;
    transfHammingWindow.resize(_FFTLength);
    for (int i=0; i <_FFTLength/2; ++i) {
      transfHammingWindow.push_back(transfHammingWindow[ind--]);
    }

    for (int j=0; j<_FFTLength; j++) {
      // Perform thresholding
      const double squaredBin = squaredModule(transfHammingWindow[j]);
      if (squaredBin <= squareThreshold) continue;

      // Insert non-zero position indexes
      sk->i.push_back(j);
      sk->j.push_back(k);

      // Take conjugate, normalize and add to array sparseKernel
      sk->real.push_back(((complex<double>) transfHammingWindow[j]).real()/_FFTLength);
      sk->imag.push_back(-((complex<double>) transfHammingWindow[j]).imag()/_FFTLength);
    }
  }
  _sparseKernel = sk;
}
