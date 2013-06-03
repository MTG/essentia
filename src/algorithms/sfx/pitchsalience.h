/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_PITCHSALIENCE_H
#define ESSENTIA_PITCHSALIENCE_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class PitchSalience : public Algorithm {

 protected:
  // parameters
  Real _lowBoundary, _highBoundary, _sampleRate;

  Input<std::vector<Real> > _spectrum;
  Output<Real> _pitchSalience;

  Algorithm* _autoCorrelation;

 public:
  PitchSalience() {
    declareInput(_spectrum, "spectrum", "the input audio spectrum");
    declareOutput(_pitchSalience, "pitchSalience", "the pitch salience (normalized from 0 to 1)");

    _autoCorrelation = AlgorithmFactory::create("AutoCorrelation");
  }

  ~PitchSalience() {
    if (_autoCorrelation) delete _autoCorrelation;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("lowBoundary", "from which frequency we are looking for the maximum (must not be larger than highBoundary) [Hz]", "(0,inf)", 100.0);
    declareParameter("highBoundary", "until which frequency we are looking for the minimum (must be smaller than half sampleRate) [Hz]", "(0,inf)", 5000.0);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class PitchSalience : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _pitchSalience;

 public:
  PitchSalience() {
    declareAlgorithm("PitchSalience");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_pitchSalience, TOKEN, "pitchSalience");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_PITCHSALIENCE_H
