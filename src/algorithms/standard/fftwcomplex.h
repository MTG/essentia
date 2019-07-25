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

#ifndef ESSENTIA_FFTWCOMPLEX_H
#define ESSENTIA_FFTWCOMPLEX_H

#include "algorithm.h"
#include "threading.h"
#include <complex>
#include <fftw3.h>

namespace essentia {
namespace standard {

class FFTWComplex : public Algorithm {

 protected:
  Input<std::vector<std::complex<Real> > >  _signal;
  Output<std::vector<std::complex<Real> > > _fft;

 public:
  FFTWComplex() : _fftPlan(0), _input(0), _output(0) {
    declareInput(_signal, "frame", "the input frame (complex)");
    declareOutput(_fft, "fft", "the FFT of the input frame");
  }

  ~FFTWComplex();

  void declareParameters() {
    declareParameter("size", "the expected size of the input frame. This is purely optional and only targeted at optimizing the creation time of the FFT object", "[1,inf)", 1024);
    declareParameter("negativeFrequencies", "returns the full spectrum or just the positive frequencies", "{true,false}", false);

  }

  void compute();
  void configure();

  static const char* name;
  static const char* category;
  static const char* description;

 protected:
  fftwf_plan _fftPlan;
  int _fftPlanSize;
  std::complex<Real>* _input;
  std::complex<Real>* _output;

  bool _negativeFrequencies;

  void createFFTObject(int size);
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class FFTWComplex : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<std::complex<Real> > > _signal;
  Source<std::vector<std::complex<Real> > > _fft;

 public:
  FFTWComplex() {
    declareAlgorithm("FFTC");
    declareInput(_signal, TOKEN, "frame");
    declareOutput(_fft, TOKEN, "fft");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_FFTWCOMPLEX_H
