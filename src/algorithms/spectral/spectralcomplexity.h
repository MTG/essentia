/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_SPECTRALCOMPLEXITY_H
#define ESSENTIA_SPECTRALCOMPLEXITY_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class SpectralComplexity : public Algorithm {

 private:
  Input<std::vector<Real> > _spectrum;
  Output<Real> _spectralComplexity;

  Algorithm* _spectralPeaks;

 public:
  SpectralComplexity() {
    declareInput(_spectrum, "spectrum", "the input spectrum");
    declareOutput(_spectralComplexity, "spectralComplexity", "the spectral complexity of the input spectrum");

    _spectralPeaks = AlgorithmFactory::create("SpectralPeaks");
  }

  ~SpectralComplexity() {
    delete _spectralPeaks;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the audio sampling rate [Hz]", "(0,inf)", 44100.);
    declareParameter("magnitudeThreshold", "the minimum spectral-peak magnitude that contributes to spectral complexity", "[0,inf)", 0.005);
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

class SpectralComplexity : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _spectralComplexity;

 public:
  SpectralComplexity() {
    declareAlgorithm("SpectralComplexity");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_spectralComplexity, TOKEN, "spectralComplexity");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_SPECTRALCOMPLEXITY_H
