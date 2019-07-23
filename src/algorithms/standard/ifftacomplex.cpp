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

#include "ifftacomplex.h"
#include "fftacomplex.h"
#include "ffta.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* IFFTAComplex::name = "IFFTC";
const char* IFFTAComplex::category = "Standard";
const char* IFFTAComplex::description = DOC("This algorithm calculates the inverse short-term Fourier transform (STFT) of an array of complex values using the FFT algorithm. The resulting frame has a size equal to the input fft frame size. The inverse Fourier transform is not defined for frames which size is less than 2 samples. Otherwise an exception is thrown.\n"
"\n"
"An exception is thrown if the input's size is not larger than 1.\n"
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


IFFTAComplex::~IFFTAComplex() {
  ForcedMutexLocker lock(FFTA::globalFFTAMutex);
  vDSP_destroy_fftsetup(fftSetup);
  delete[] accelBuffer.realp;
  delete[] accelBuffer.imagp;
}

void IFFTAComplex::compute() {

  const std::vector<std::complex<Real> >& fft = _fft.get();
  std::vector<std::complex<Real> >& signal = _signal.get();

  // check if input is OK
  int size = (int)fft.size();
  if (size <= 0) {
    throw EssentiaException("IFFTC: Input size cannot be 0 or 1");
  }
  if ((fftSetup == 0) ||
      ((fftSetup != 0) && _fftPlanSize != size)) {
    createFFTObject(size);
  }

  //Pack
  // accelBuffer.realp[0] = fft[0].real();
  // accelBuffer.imagp[0] = fft[fft.size()-1].real();
    
  for(int i = 0; i < fft.size(); i++) {
      accelBuffer.realp[i] = fft[i].real();
      accelBuffer.imagp[i] = fft[i].imag();
  }
    
  vDSP_fft_zip(fftSetup, &accelBuffer, 1, logSize, FFT_INVERSE);
  
  // Copy result from plan to output vector.
  signal.resize(size);
  
  vDSP_ztoc(&accelBuffer, 1, (COMPLEX*)&signal[0], 2, size);

  if (_normalize) {
    Real norm = (Real)size;
    
    for (int i = 0; i < size; i++) {
      signal[i] /= norm;
    }
  }
}

void IFFTAComplex::configure() {
  createFFTObject(parameter("size").toInt());
  _normalize = parameter("normalize").toBool();
}

void IFFTAComplex::createFFTObject(int size) {
  ForcedMutexLocker lock(FFTA::globalFFTAMutex);
    
  // Delete stuff before assigning.
  delete[] accelBuffer.realp;
  delete[] accelBuffer.imagp;

  accelBuffer.realp = new float[size];
  accelBuffer.imagp = new float[size];

  logSize = log2(size);
    
  // With vDSP you only need to create a new fft if you've increased the size.
  if(size > _fftPlanSize) {
    vDSP_destroy_fftsetup(fftSetup);
    fftSetup = vDSP_create_fftsetup(logSize, 0);
  }

  _fftPlanSize = size;
}
