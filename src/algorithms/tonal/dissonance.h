/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_DISSONANCE_H
#define ESSENTIA_DISSONANCE_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Dissonance : public Algorithm {

 protected:
  Input<std::vector<Real> > _frequencies;
  Input<std::vector<Real> > _magnitudes;
  Output<Real> _dissonance;

 public:
  Dissonance() {
    declareInput(_frequencies, "frequencies", "the frequencies of the spectral peaks (must be sorted by frequency)");
    declareInput(_magnitudes, "magnitudes", "the magnitudes of the spectral peaks (must be sorted by frequency");
    declareOutput(_dissonance, "dissonance", "the dissonance of the audio signal (0 meaning completely consonant, and 1 meaning completely dissonant)");
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Dissonance : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _frequencies;
  Sink<std::vector<Real> > _magnitudes;
  Source<Real> _dissonance;

 public:
  Dissonance() {
    declareAlgorithm("Dissonance");
    declareInput(_frequencies, TOKEN, "frequencies");
    declareInput(_magnitudes, TOKEN, "magnitudes");
    declareOutput(_dissonance, TOKEN, "dissonance");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_DISSONANCE_H
