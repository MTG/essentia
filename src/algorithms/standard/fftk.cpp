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

#include "fftk.h"
#include "essentia.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* FFTK::name = "FFT";
const char* FFTK::category = "Standard";
const char* FFTK::description = DOC("This algorithm computes the positive complex short-term Fourier transform (STFT) of an array using the FFT algorithm.  The resulting fft has a size of (s/2)+1, where s is the size of the input frame.\n"
"At the moment FFT can only be computed on frames which size is even and non zero, otherwise an exception is thrown.\n"
"\n"
"FFT computation will be carried out using the KISS library [3]"
"\n"
"References:\n"
"  [1] Fast Fourier transform - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Fft\n\n"
"  [2] Fast Fourier Transform -- from Wolfram MathWorld,\n"
"  http://mathworld.wolfram.com/FastFourierTransform.html\n"
"  [3] KISS -- Keep It Simple, Stupid.\n"
"  http://kissfft.sourceforge.net/");

ForcedMutex FFTK::globalFFTKMutex;

FFTK::~FFTK() {
  ForcedMutexLocker lock(globalFFTKMutex);

  // we might have called essentia::shutdown() before this algorithm goes out
  // of scope, so make sure we're not doing stupid things here
  // This will cause a memory leak then, but it is definitely a better choice
  // than a crash (right, right??? :-) )
  if (essentia::isInitialized()) {
    free(_fftCfg);
    free(_input);
    free(_output);
  }
}

void FFTK::compute() {

  const std::vector<Real>& signal = _signal.get();
  std::vector<std::complex<Real> >& fft = _fft.get();

  // check if input is OK
  int size = int(signal.size());
  if (size == 0) {
    throw EssentiaException("FFT: Input size cannot be 0");
  }

  if (_fftCfg == 0 || (_fftCfg != 0 && _fftPlanSize != size)) {
    createFFTObject(size);
  }

  // copy input into plan
  memcpy(_input, &signal[0], size*sizeof(Real));
    
  kiss_fftr(_fftCfg, (kiss_fft_scalar *) _input, (kiss_fft_cpx *) _output);

  // copy result from plan to output vector
  fft.resize(size/2+1);
  memcpy(&fft[0], _output, (size/2+1)*sizeof(complex<Real>));
}

void FFTK::configure() {
  createFFTObject(parameter("size").toInt());
}

void FFTK::createFFTObject(int size) {
  ForcedMutexLocker lock(globalFFTKMutex);

  // This is only needed because at the moment we return half of the spectrum,
  // which means that there are 2 different input signals that could yield the
  // same FFT...
  if (size % 2 == 1) {
    throw EssentiaException("FFT: can only compute FFT of arrays which have an even size");
  }

  // create the temporary storage array
  free(_input);
  free(_output);
  _input = (Real*)malloc(sizeof(Real)*size);
  _output = (complex<Real>*)malloc(sizeof(complex<Real>)*size);

    
  if (_fftCfg != 0) {
    free(_fftCfg);
  }
    
  _fftCfg = kiss_fftr_alloc(size, 0, NULL, NULL );    
  _fftPlanSize = size;
}
