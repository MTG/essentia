/*
 * Copyright (C) 2006-2015  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_STOCHASTICMODELSYNTH_H
#define ESSENTIA_STOCHASTICMODELSYNTH_H

#include "algorithm.h"
#include "algorithmfactory.h"
#include <fstream>


namespace essentia {
namespace standard {

class StochasticModelSynth : public Algorithm {

 protected:

  Input<std::vector<Real> > _stocenv;
  Output<std::vector<Real> > _frame;

  Real _stocf;
  int _fftSize;
  int _stocSize;
  int _hopSize;
  int _hN ; // half fftsize

  Algorithm* _window;
  Algorithm* _ifft;
  Algorithm* _resample;
  Algorithm* _overlapadd;


 public:
  StochasticModelSynth() {

    declareInput(_stocenv, "stocenv", "the stochastic envelope input");
    declareOutput(_frame, "frame", "the output frame");

    _resample = AlgorithmFactory::create("ResampleFFT");
    _ifft = AlgorithmFactory::create("IFFT");
    _window = AlgorithmFactory::create("Windowing");
    _overlapadd = AlgorithmFactory::create("OverlapAdd");

  }

  ~StochasticModelSynth() {

  delete _window;
  delete _ifft;
  delete _overlapadd;
  delete _resample;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("hopSize", "the hop size between frames", "[1,inf)", 512);
    declareParameter("fftSize", "the size of the internal FFT size (full spectrum size)", "[1,inf)", 2048);
    declareParameter("stocf", "decimation factor used for the stochastic approximation", "(0,1]", 0.2);

  }

  void configure();
  void compute();

  void getFFTFromEnvelope(const std::vector<Real> magResDB, std::vector<std::complex<Real> > &fftStoc);

  static const char* name;
  static const char* category;
  static const char* description;



 private:

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class StochasticModelSynth : public StreamingAlgorithmWrapper {

 protected:

  Sink<std::vector<Real> > _stocenv;
  Source<std::vector<Real> > _frame; // output

 public:
  StochasticModelSynth() {
    declareAlgorithm("StochasticModelSynth");
    declareInput(_stocenv, TOKEN, "stocenv");
    declareOutput(_frame, TOKEN, "frame");
  }
};

} // namespace streaming
} // namespace essentia




#endif // ESSENTIA_STOCHASTICMODELSYNTH_H
