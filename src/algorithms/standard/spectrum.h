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

#ifndef ESSENTIA_SPECTRUM_H
#define ESSENTIA_SPECTRUM_H

#include "algorithmfactory.h"
#include <complex>

namespace essentia {
namespace standard {

class Spectrum : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _spectrum;

  Algorithm* _fft;
  Algorithm* _magnitude;
  std::vector<std::complex<Real> > _fftBuffer;

 public:
  Spectrum() {
    declareInput(_signal, "frame", "the input audio frame");
    declareOutput(_spectrum, "spectrum", "the magnitude spectrum of the input audio signal");

    _fft = AlgorithmFactory::create("FFT");
    _magnitude = AlgorithmFactory::create("Magnitude");
  }

  ~Spectrum() {
    delete _fft;
    delete _magnitude;
  }

  void declareParameters() {
    declareParameter("size", "the expected size of the input audio signal (this is an optional parameter to optimize memory allocation)", "[1,inf)", 2048);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Spectrum : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<std::vector<Real> > _spectrum;

 public:
  Spectrum() {
    declareAlgorithm("Spectrum");
    declareInput(_signal, TOKEN, "frame");
    declareOutput(_spectrum, TOKEN, "spectrum");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_SPECTRUM_H
