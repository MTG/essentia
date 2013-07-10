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

#ifndef ESSENTIA_FFTW_H
#define ESSENTIA_FFTW_H

#include "algorithm.h"
#include "threading.h"
#include <complex>
#include <fftw3.h>

namespace essentia {
namespace standard {

class FFTW : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<std::complex<Real> > > _fft;

 public:
  FFTW() : _fftPlan(0), _input(0), _output(0) {
    declareInput(_signal, "frame", "the input audio frame");
    declareOutput(_fft, "fft", "the FFT of the input frame");
  }

  ~FFTW();

  void declareParameters() {
    declareParameter("size", "the expected size of the input frame. This is purely optional and only targeted at optimizing the creation time of the FFT object", "[1,inf)", 1024);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;

 protected:
  friend class IFFTW;
  static ForcedMutex globalFFTWMutex;

  fftwf_plan _fftPlan;
  int _fftPlanSize;
  Real* _input;
  std::complex<Real>* _output;

  void createFFTObject(int size);
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class FFTW : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<std::vector<std::complex<Real> > > _fft;

 public:
  FFTW() {
    declareAlgorithm("FFT");
    declareInput(_signal, TOKEN, "frame");
    declareOutput(_fft, TOKEN, "fft");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_FFTW_H
