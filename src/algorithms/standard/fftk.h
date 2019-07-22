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

#ifndef ESSENTIA_FFTK_H
#define ESSENTIA_FFTK_H

#include "algorithm.h"
#include "threading.h"
#include <complex>
#include "tools/kiss_fftr.h"

namespace essentia {
namespace standard {

class FFTK : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<std::complex<Real> > > _fft;

 public:
  FFTK() : _input(0), _output(0), _fftCfg(0) {
    declareInput(_signal, "frame", "the input audio frame");
    declareOutput(_fft, "fft", "the FFT of the input frame");
  }

  ~FFTK();

  void declareParameters() {
    declareParameter("size", "the expected size of the input frame. This is purely optional and only targeted at optimizing the creation time of the FFT object", "[1,inf)", 1024);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* category;
  static const char* description;

 protected:
  friend class IFFTK;
  friend class FFTKComplex;
  friend class IFFTKComplex;

  static ForcedMutex globalFFTKMutex;
   
  int _fftPlanSize;
  Real* _input;
  std::complex<Real>* _output;
        
  kiss_fftr_cfg _fftCfg;

  void createFFTObject(int size);
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class FFTK : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<std::vector<std::complex<Real> > > _fft;

 public:
  FFTK() {
    declareAlgorithm("FFT");
    declareInput(_signal, TOKEN, "frame");
    declareOutput(_fft, TOKEN, "fft");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_FFTK_H
