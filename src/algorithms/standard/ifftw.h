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

#ifndef ESSENTIA_IFFTW_H
#define ESSENTIA_IFFTW_H

#include "algorithm.h"
#include "threading.h"
#include <complex>
#include <fftw3.h>

namespace essentia {
namespace standard {

class IFFTW : public Algorithm {

 protected:
  Input<std::vector<std::complex<Real> > > _fft;
  Output<std::vector<Real> > _signal;

 public:
  IFFTW() : _fftPlan(0), _input(0), _output(0) {
    declareInput(_fft, "fft", "the input frame");
    declareOutput(_signal, "frame", "the IFFT of the input frame");
  }

  ~IFFTW();

  void declareParameters() {
    declareParameter("size", "the expected size of the input frame. This is purely optional and only targeted at optimizing the creation time of the FFT object", "[1,inf)", 1024);
  }


  void compute();
  void configure();

  static const char* name;
  static const char* description;

 protected:
  fftwf_plan _fftPlan;
  int _fftPlanSize;
  std::complex<Real>* _input;
  Real* _output;

  void createFFTObject(int size);
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class IFFTW : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<std::complex<Real> > > _fft;
  Source<std::vector<Real> > _signal;

 public:
  IFFTW() {
    declareAlgorithm("IFFT");
    declareInput(_fft, TOKEN, "fft");
    declareOutput(_signal, TOKEN, "frame");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_IFFTW_H
