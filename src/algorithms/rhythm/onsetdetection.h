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

#ifndef ESSENTIA_ONSETDETECTION_H
#define ESSENTIA_ONSETDETECTION_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class OnsetDetection : public Algorithm {

 private:
  Input<std::vector<Real> > _spectrum;
  Input<std::vector<Real> > _phase;
  Output<Real> _onsetDetection;

  Algorithm* _hfc;
  Algorithm* _flux;
  Algorithm* _melBands;
  std::string _method;

 public:
  OnsetDetection() {
    declareInput(_spectrum, "spectrum", "the input spectrum");
    declareInput(_phase, "phase", "the phase vector corresponding to this spectrum--used only by the \"complex\" method");
    declareOutput(_onsetDetection, "onsetDetection", "the value of the detection function in the current frame");

    _hfc = AlgorithmFactory::create("HFC");
    _flux = AlgorithmFactory::create("Flux");
    _melBands = AlgorithmFactory::create("MelBands");
  }

  ~OnsetDetection() {
    if (_hfc) delete _hfc;
    if (_flux) delete _flux;
    if (_melBands) delete _melBands;
  }

  void declareParameters() {
    declareParameter("method", "the method used for onset detection", "{hfc,complex,complex_phase,flux,melflux,rms}", "hfc");
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.0);
  }

  void reset();
  void configure();
  void compute();

  static const char* name;
  static const char* description;

  std::vector<Real> _phase_1;
  std::vector<Real> _phase_2;
  std::vector<Real> _spectrum_1;
  Real _rmsOld;
  bool _firstFrame;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class OnsetDetection : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _phase;
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _onsetDetection;

 public:
  OnsetDetection() {
    declareAlgorithm("OnsetDetection");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareInput(_phase, TOKEN, "phase");
    declareOutput(_onsetDetection, TOKEN, "onsetDetection");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_ONSETDETECTION_H
