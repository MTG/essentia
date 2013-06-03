/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_ENVELOPE_H
#define ESSENTIA_ENVELOPE_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Envelope : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _envelope;

 public:
  Envelope() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_envelope, "signal", "the resulting envelope of the signal");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the audio sampling rate [Hz]", "(0,inf)", 44100.);
    declareParameter("attackTime", "the attack time of the first order lowpass in the attack phase [ms]", "[0,inf)", 10.0);
    declareParameter("releaseTime", "the release time of the first order lowpass in the release phase [ms]", "[0,inf)", 1500.0);
    declareParameter("applyRectification", "whether to apply rectification (envelope based on the absolute value of signal)", "{true,false}", true);
  }

  void configure();
  void reset();
  void compute();

  static const char* name;
  static const char* description;

 protected:
  // output of the filter
  Real _tmp;

  // attack and release coefficient for the filter
  Real _ga;
  Real _gr;

  bool _applyRectification;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Envelope : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _signal;
  Source<Real> _envelope;

 public:
  Envelope() {
    declareAlgorithm("Envelope");
    declareInput(_signal, STREAM, 4096, "signal");
    declareOutput(_envelope, STREAM, 4096, "signal");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_ENVELOPE_H
