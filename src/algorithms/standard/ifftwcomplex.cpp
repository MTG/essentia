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

#include "ifftwcomplex.h"
#include "fftw.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* IFFTWComplex::name = "IFFTC";
const char* IFFTWComplex::category = "Standard";
const char* IFFTWComplex::description = DOC("This algorithm calculates the inverse short-term Fourier transform (STFT) of an array of complex values using the FFT algorithm. The resulting frame has a size equal to the input fft frame size. The inverse Fourier transform is not defined for frames which size is less than 2 samples. Otherwise an exception is thrown.\n"
"\n"
"An exception is thrown if the input's size is not larger than 1.\n"
"\n"
"References:\n"
"  [1] Fast Fourier transform - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Fft\n\n"
"  [2] Fast Fourier Transform -- from Wolfram MathWorld,\n"
"  http://mathworld.wolfram.com/FastFourierTransform.html");


IFFTWComplex::~IFFTWComplex() {
  ForcedMutexLocker lock(FFTW::globalFFTWMutex);

  fftwf_destroy_plan(_fftPlan);
  fftwf_free(_input);
  fftwf_free(_output);
}

void IFFTWComplex::compute() {

  const std::vector<std::complex<Real> >& fft = _fft.get();
  std::vector<std::complex<Real> >& signal = _signal.get();

  // check if input is OK
  int size = (int)fft.size();
  if (size <= 0) {
    throw EssentiaException("IFFTComplex: Input size cannot be 0 or 1");
  }
  if ((_fftPlan == 0) ||
      ((_fftPlan != 0) && _fftPlanSize != size)) {
    createFFTObject(size);
  }

  // copy input into plan
  memcpy(_input, &fft[0], size*sizeof(complex<Real>));

  // calculate the fft
  fftwf_execute(_fftPlan);

  // copy result from plan to output vector
  signal.resize(size);
  memcpy(&signal[0], _output, size*sizeof(complex<Real>));

  if (_normalize) {
    Real norm = (Real)size;
    
    for (int i = 0; i < size; i++) {
      signal[i] /= norm;
    }
  }
}

void IFFTWComplex::configure() {
  createFFTObject(parameter("size").toInt());
  _normalize = parameter("normalize").toBool();
}

void IFFTWComplex::createFFTObject(int size) {
  ForcedMutexLocker lock(FFTW::globalFFTWMutex);

  // create the temporary storage array
  fftwf_free(_input);
  fftwf_free(_output);
  _input = (complex<Real>*)fftwf_malloc(sizeof(complex<Real>)*size);
  _output = (complex<Real>*)fftwf_malloc(sizeof(complex<Real>)*size);

  if (_fftPlan != 0) {
    fftwf_destroy_plan(_fftPlan);
  }

  //_fftPlan = fftwf_plan_dft_c2r_1d(size, (fftwf_complex*)_input, _output, FFTW_MEASURE);
  _fftPlan = fftwf_plan_dft_1d(size, (fftwf_complex*)_input, (fftwf_complex*)_output, FFTW_BACKWARD, FFTW_ESTIMATE);
  _fftPlanSize = size;

}

