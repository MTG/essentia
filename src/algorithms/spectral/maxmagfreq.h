/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_MAXMAGFREQ_H
#define ESSENTIA_MAXMAGFREQ_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class MaxMagFreq : public Algorithm {

 private:
  Input<std::vector<Real> > _spectrum;
  Output<Real> _maxMagFreq;
  Real _sampleRate;

 public:
  MaxMagFreq() {
    declareInput(_spectrum, "spectrum", "the input spectrum (must have more than 1 element)");
    declareOutput(_maxMagFreq, "maxMagFreq", "the frequency with the largest magnitude [Hz]");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the audio sampling rate [Hz]", "(0,inf)", 44100.);
  }

  void configure() {
    _sampleRate = parameter("sampleRate").toReal();
  }

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class MaxMagFreq : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _maxMagFreq;

 public:
  MaxMagFreq() {
    declareAlgorithm("MaxMagFreq");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_maxMagFreq, TOKEN, "maxMagFreq");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_MAXMAGFREQ_H
