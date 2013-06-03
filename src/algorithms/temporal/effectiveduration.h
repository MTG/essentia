/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
