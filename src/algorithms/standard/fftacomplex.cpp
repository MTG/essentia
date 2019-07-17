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

#include "fftacomplex.h"
#include "ffta.h"
#include "essentia.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* FFTAComplex::name = "FFTC";
const char* FFTAComplex::category = "Standard";
const char* FFTAComplex::description = DOC("This algorithm computes the complex short-term Fourier transform (STFT) of a complex array using the FFT algorithm. If the `negativeFrequencies` flag is set on, the resulting fft has a size of (s/2)+1, where s is the size of the input frame. Otherwise, output matches the input size.\n"
"At the moment FFT can only be computed on frames which size is even and non zero, otherwise an exception is thrown.\n"
"\n"
"FFT computation will be carried out using the Accelerate Framework [3]"
"\n"
"References:\n"
"  [1] Fast Fourier transform - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Fft\n\n"
"  [2] Fast Fourier Transform -- from Wolfram MathWorld,\n"
"  http://mathworld.wolfram.com/FastFourierTransform.html\n"
"  [3] vDSP Programming Guide -- from Apple\n"
"  https://developer.apple.com/library/ios/documentation/Performance/Conceptual/vDSP_Programming_Guide/UsingFourierTransforms/UsingFourierTransforms.html"
);

//ForcedMutex FFTAComplex::globalFFTAComplexMutex;

FFTAComplex::~FFTAComplex() {
  ForcedMutexLocker lock(FFTA::globalFFTAMutex);

  // we might have called essentia::shutdown() before this algorithm goes out
  // of scope, so make sure we're not doing stupid things here
  // This will cause a memory leak then, but it is definitely a better choice
  // than a crash (right, right??? :-) )
  if (essentia::isInitialized()) {
    vDSP_destroy_fftsetup(fftSetup);
    delete[] accelBuffer.realp;
    delete[] accelBuffer.imagp;
  }
}

void FFTAComplex::compute() {
  const std::vector<std::complex<Real> >& signal = _signal.get();
  std::vector<std::complex<Real> >& fft = _fft.get();

  // check if input is OK
  int size = int(signal.size());
  if (size == 0) {
    throw EssentiaException("FFTC: Input size cannot be 0");
  }
 
  if ((fftSetup == 0) ||
      ((fftSetup != 0) && _fftPlanSize != size)) {
    createFFTObject(size);
  }

  // Copy the interleaved complex vector to a split complex vector.
  vDSP_ctoz((DSPComplex*)&signal[0], 2, &accelBuffer, 1, _fftPlanSize);
  // Do complex->complex forward FFT
  vDSP_fft_zip(fftSetup, &accelBuffer, 1, logSize, FFT_FORWARD);

  fft.resize(_fftOutSize);

  //Construct first point
  // fft[0] = std::complex<Real>(accelBuffer.realp[0]/2.0f, 0.0f);
    
  for(int i=0; i<_fftOutSize; i++) {
    std::complex<Real> point(accelBuffer.realp[i], accelBuffer.imagp[i]);
    fft[i] = point;
  }
    
  //Construct the last point
  // fft[size/2] = std::complex<Real>(accelBuffer.imagp[0]/2.0f, 0.0f);
}

void FFTAComplex::configure() {
  _negativeFrequencies = parameter("negativeFrequencies").toBool();
  createFFTObject(parameter("size").toInt());
}

void FFTAComplex::createFFTObject(int size) {
  ForcedMutexLocker lock(FFTA::globalFFTAMutex);

  // This is only needed because at the moment we return half of the spectrum,
  // which means that there are 2 different input signals that could yield the
  // same FFT...
  if (size % 2 == 1) {
    throw EssentiaException("FFTC: can only compute FFTC of arrays which have an even size");
  }

  if (_negativeFrequencies) {
    _fftOutSize = size;
  }
  else {
    _fftOutSize = size / 2 + 1;
  }

  logSize = log2(size);

  delete[] accelBuffer.realp;
  delete[] accelBuffer.imagp;
  accelBuffer.realp = new float[size];
  accelBuffer.imagp = new float[size];

  // With vDSP you only need to create a new fft if you've increased the size
  if(size > _fftPlanSize) {
    vDSP_destroy_fftsetup(fftSetup);
    fftSetup = vDSP_create_fftsetup(logSize, 0);
  }

  _fftPlanSize = size;
}
