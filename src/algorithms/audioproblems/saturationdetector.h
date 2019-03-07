/*
 * Copyright (C) 2006-2019  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef SATURATIONDETECTEOR_H
#define SATURATIONDETECTEOR_H

#include "algorithm.h"
#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class SaturationDetector : public Algorithm {
 private:
  Input<std::vector<Real> > _frame;
  Output<std::vector<Real>> _starts;
  Output<std::vector<Real>> _ends;

  Real _sampleRate;
  uint _hopSize, _frameSize;
  Real _minimumDuration;
  Real _energyThreshold, _differentialThreshold;
  uint _idx;
  Real _previousStart;
  uint _startProc, _endProc;

 public:
  SaturationDetector() {
    declareInput(_frame, "frame", "the input audio frame");
    declareOutput(_starts, "starts", "starting times of the detected saturated regions [s]");
    declareOutput(_ends, "ends", "ending times of the detected saturated regions [s]");
  }

  void declareParameters() {
    declareParameter("sampleRate", "sample rate used for the analysis", "(0,inf)", 44100.);
    declareParameter("frameSize", "expected input frame size", "(0,inf)", 512);
    declareParameter("hopSize", "hop size used for the analysis", "(0,inf)", 256);
    declareParameter("energyThreshold", "mininimum energy of the samples in the saturated regions [dB]", "(-inf,0])", -1.f);
    declareParameter("differentialThreshold", "minimum difference between contiguous samples of the salturated regions", "[0,inf))", 0.001f);
    declareParameter("minimumDuration", "minimum duration of the saturated regions [ms]", "[0,inf))", 0.005f);
  }

  void configure();
  void compute();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace standard
} // namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class SaturationDetector : public StreamingAlgorithmWrapper {
 protected:
  Sink<std::vector<Real> > _frame;
  Source<std::vector<Real> > _starts;
  Source<std::vector<Real> > _ends;

 public:
  SaturationDetector() {
    declareAlgorithm("SaturationDetector");
    declareInput(_frame, TOKEN, "frame");
    declareOutput(_starts, TOKEN, "starts");
    declareOutput(_ends, TOKEN, "ends");
  }
};

} // namespace streaming
} // namespace essentia

#endif // SATURATIONDETECTEOR_H
