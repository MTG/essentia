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

#ifndef ESSENTIA_LOUDNESSVICKERS_H
#define ESSENTIA_LOUDNESSVICKERS_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class LoudnessVickers : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<Real> _loudness;

  Real _sampleRate;
  Real _Vms;
  Real _c;
  Algorithm* _filtering;

 public:
  LoudnessVickers() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_loudness, "loudness", "the Vickers loudness [dB]");

    _filtering = AlgorithmFactory::create("IIR");
  }

  ~LoudnessVickers() {
    if (_filtering) delete _filtering;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the audio sampling rate of the input signal which is used to create the weight vector [Hz] (currently, this algorithm only works on signals with a sampling rate of 44100Hz)", "[44100,44100]", 44100.);
  }

  void configure();
  void compute();

  void reset() {
    _filtering->reset();
    _Vms = 0.0;
  }

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class LoudnessVickers : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<Real> _loudness;

 public:
  LoudnessVickers() {
    declareAlgorithm("LoudnessVickers");
    declareInput(_signal, TOKEN, "signal");
    declareOutput(_loudness, TOKEN, "loudness");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_LOUDNESSVICKERS_H
