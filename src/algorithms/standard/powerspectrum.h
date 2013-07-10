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

#ifndef ESSENTIA_POWERSPECTRUM_H
#define ESSENTIA_POWERSPECTRUM_H

#include "algorithmfactory.h"
#include <complex>

namespace essentia {
namespace standard {

class PowerSpectrum : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _powerSpectrum;

  Algorithm* _fft;
  std::vector<std::complex<Real> > _fftBuffer;

 public:
  PowerSpectrum() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_powerSpectrum, "powerSpectrum", "the power spectrum of the input signal");

    // creation of the FFT algorithm
    _fft = AlgorithmFactory::create("FFT");
  }

  ~PowerSpectrum() {
    delete _fft;
  }

  void declareParameters() {
    declareParameter("size", "the expected size of the input frame (this is purely optional and only targeted at optimizing the creation time of the FFT object)", "[1,inf)", 1024);
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

class PowerSpectrum : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<std::vector<Real> > _powerSpectrum;

 public:
  PowerSpectrum() {
    declareAlgorithm("PowerSpectrum");
    declareInput(_signal, TOKEN, "signal");
    declareOutput(_powerSpectrum, TOKEN, "powerSpectrum");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_POWERSPECTRUM_H
