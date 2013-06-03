/*
 * Copyright (C) 2006-2009 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_TRISTIMULUS_H
#define ESSENTIA_TRISTIMULUS_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Tristimulus : public Algorithm {

 protected:
  Input<std::vector<Real> > _frequencies;
  Input<std::vector<Real> > _magnitudes;
  Output<std::vector<Real> > _tristimulus;

 public:
  Tristimulus() {
    declareInput(_frequencies, "frequencies", "the frequencies of the harmonic peaks ordered by frequency");
    declareInput(_magnitudes, "magnitudes", "the magnitudes of the harmonic peaks ordered by frequency");
    declareOutput(_tristimulus, "tristimulus", "a three-element vector that measures the mixture of harmonics of the given spectrum");
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

class Tristimulus : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _frequencies;
  Sink<std::vector<Real> > _magnitudes;
  Source<std::vector<Real> > _tristimulus;

 public:
  Tristimulus() {
    declareAlgorithm("Tristimulus");
    declareInput(_frequencies, TOKEN, "frequencies");
    declareInput(_magnitudes, TOKEN, "magnitudes");
    declareOutput(_tristimulus, TOKEN, "tristimulus");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_TRISTIMULUS_H
