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

#include "afft.h"
#include "essentia.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* AFFT::name = "FFT";
const char* AFFT::description = DOC("AFFT Description");

ForcedMutex AFFT::globalFFTWMutex;

AFFT::~AFFT() {
  ForcedMutexLocker lock(globalFFTWMutex);

  // we might have called essentia::shutdown() before this algorithm goes out
  // of scope, so make sure we're not doing stupid things here
  // This will cause a memory leak then, but it is definitely a better choice
  // than a crash (right, right??? :-) )
  if (essentia::isInitialized()) {
//    fftwf_destroy_plan(_fftPlan);
//    fftwf_free(_input);
//    fftwf_free(_output);
  }
}

void AFFT::compute() {
    
  const std::vector<Real>& signal = _signal.get();
  std::vector<std::complex<Real> >& fft = _fft.get();

  // check if input is OK
  int size = int(signal.size());
  if (size == 0) {
    throw EssentiaException("FFT: Input size cannot be 0");
  }
 
//  if ((_fftPlan == 0) ||
//      ((_fftPlan != 0) && _fftPlanSize != size)) {
//    createFFTObject(size);
//  }
    
        createFFTObject(size);
    

    for(int i=0; i<size*sizeof(Real); i++)
    {
//        std::cout << "signal: " << signal[i] << "\n";
    }
    
  // copy input into plan
    
    _input = (Real *) malloc(size*sizeof(Real));
    memcpy(_input, &signal[0], size*sizeof(Real));
    
    
  // calculate the fft
//  fftwf_execute(_fftPlan);
    
    DSPSplitComplex A;
    A.realp = new float[size/2];
    A.imagp = new float[size/2];
    
    
    // Scramble-pack the real data into complex buffer in just the way that's
    // required by the real-to-complex FFT function that follows.
    vDSP_ctoz((DSPComplex*)_input, 2, &A, 1, size/2);
    
    // Do real->complex forward FFT
    vDSP_fft_zrip(fftSetup, &A, 1, logSize, FFT_FORWARD);
    
    //Scale down by /2.0f

    fft.resize(size/2+1);
    
    fft[0] = std::complex<Real>(A.realp[0]/2.0f, 0.0f);
    
    for(int i=1; i<size/2; i++) {
        std::complex<Real> bin(A.realp[i]/2.0f, A.imagp[i]/2.0f);
        
        fft[i] = bin;
    }
    
    fft.push_back(std::complex<Real>(A.imagp[0]/2.0f, 0.0f));
    
//    fft[i]


  // copy result from plan to output vector
    

//  memcpy(&fft[0], _output, (size/2+1)*sizeof(complex<Real>));
}

void AFFT::configure() {
  createFFTObject(parameter("size").toInt());
}

void AFFT::createFFTObject(int size) {
  ForcedMutexLocker lock(globalFFTWMutex);

  // This is only needed because at the moment we return half of the spectrum,
  // which means that there are 2 different input signals that could yield the
  // same FFT...
  if (size % 2 == 1) {
    throw EssentiaException("FFT: can only compute FFT of arrays which have an even size");
  }
    
//  // create the temporary storage array
//  fftwf_free(_input);
//  fftwf_free(_output);
//  _input = (Real*)fftwf_malloc(sizeof(Real)*size);
//  _output = (complex<Real>*)fftwf_malloc(sizeof(complex<Real>)*size);
//
//  if (_fftPlan != 0) {
//    fftwf_destroy_plan(_fftPlan);
//  }
//
//  _fftPlan = fftwf_plan_dft_r2c_1d(size, _input, (fftwf_complex*)_output, FFTW_ESTIMATE);
    
    //NEED TO STOP CREATING NEW FFTOBJECTS, CHECKFORSIZE
    std::cout << "HERE\n";
    
    logSize = log2(size);
    
    fftSetup = vDSP_create_fftsetup( logSize, 0 );
    

  _fftPlanSize = size;
}
