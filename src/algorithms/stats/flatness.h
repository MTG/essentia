/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_FLATNESS_H
#define ESSENTIA_FLATNESS_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class Flatness : public Algorithm {

 protected:
  Input<std::vector<Real> > _array;
  Output<Real> _flatness;
  Algorithm* _geometricMean;

 public:
  Flatness() {
    declareInput(_array, "array", "the input array");
    declareOutput(_flatness, "flatness", "the flatness (ratio between the geometric and the arithmetic mean of the input array)");

    _geometricMean = AlgorithmFactory::create("GeometricMean");
  }

  ~Flatness() {
    delete _geometricMean;
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

class Flatness : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<Real> _flatness;

 public:
  Flatness() {
    declareAlgorithm("Flatness");
    declareInput(_array, TOKEN, "array");
    declareOutput(_flatness, TOKEN, "flatness");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_FLATNESS_H
