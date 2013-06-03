/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_LARM_H
#define ESSENTIA_LARM_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class Larm : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<Real> _larm;
  Algorithm* _envelope;
  Algorithm* _powerMean;

 public:
  Larm() {
    declareInput(_signal, "signal", "the audio input signal");
    declareOutput(_larm, "larm", "the LARM loudness estimate [dB]");

    _envelope = AlgorithmFactory::create("Envelope");
    _powerMean = AlgorithmFactory::create("PowerMean");
  }

  ~Larm() {
    delete _envelope;
    delete _powerMean;
  }

  void reset() {
    _envelope->reset();
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("attackTime", "the attack time of the first order lowpass in the attack phase [ms]", "[0,inf)", 10.0);
    declareParameter("releaseTime", "the release time of the first order lowpass in the release phase [ms]", "[0,inf)", 1500.0);
    declareParameter("power", "the power used for averaging", "(-inf,inf)", 1.5); // 1.5 is an empirical value, see the paper...
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

class Larm : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<Real> _larm;

 public:
  Larm() {
    declareAlgorithm("Larm");
    declareInput(_signal, TOKEN, "signal");
    declareOutput(_larm, TOKEN, "larm");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_LARM_H
