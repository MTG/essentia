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

#ifndef ESSENTIA_EFFECTIVEDURATION_H
#define ESSENTIA_EFFECTIVEDURATION_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class EffectiveDuration : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<Real> _effectiveDuration;

 public:
  EffectiveDuration() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_effectiveDuration, "effectiveDuration", "the effective duration of the signal [s]");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
  }

  void compute();

  static const char* name;
  static const char* description;

  static const Real thresholdRatio;
  static const Real noiseFloor;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class EffectiveDuration : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<Real> _effectiveDuration;

 public:
  EffectiveDuration() {
    declareAlgorithm("EffectiveDuration");
    declareInput(_signal, TOKEN, "signal");
    declareOutput(_effectiveDuration, TOKEN, "effectiveDuration");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_EFFECTIVEDURATION_H
