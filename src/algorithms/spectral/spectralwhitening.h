/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef SPECTRALWHITENING_H
#define SPECTRALWHITENING_H

#include "algorithm.h"
#include "bpfutil.h"

namespace essentia {
namespace standard {

class SpectralWhitening : public Algorithm {

 protected:
  Input< std::vector<Real> > _spectrum;
  Input< std::vector<Real> > _frequencies;
  Input< std::vector<Real> > _magnitudes;
  Output< std::vector<Real> > _magnitudesWhite;

  Real _maxFreq;
  Real _spectralRange;

  essentia::util::BPF _noiseBPF;

 public:
  SpectralWhitening() {
    declareInput(_spectrum, "spectrum", "the audio linear spectrum");
    declareInput(_frequencies, "frequencies", "the spectral peaks' linear frequencies");
    declareInput(_magnitudes, "magnitudes", "the spectral peaks' linear magnitudes");
    declareOutput(_magnitudesWhite, "magnitudes", "the whitened spectral peaks' linear magnitudes");
  }

  ~SpectralWhitening() {
  }

  void declareParameters() {
    declareParameter("maxFrequency", "max frequency to apply whitening to [Hz]", "(0,inf)", 5000.0);
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;

  static const Real bpfResolution;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class SpectralWhitening : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Sink<std::vector<Real> > _frequencies;
  Sink<std::vector<Real> > _magnitudes;
  Source<std::vector<Real> > _magnitudesWhite;

 public:
  SpectralWhitening() {
    declareAlgorithm("SpectralWhitening");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareInput(_frequencies, TOKEN, "frequencies");
    declareInput(_magnitudes, TOKEN, "magnitudes");
    declareOutput(_magnitudesWhite, TOKEN, "magnitudes");
  }
};

} // namespace streaming
} // namespace essentia

#endif // SPECTRALWHITENING_H
