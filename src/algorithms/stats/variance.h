/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_VARIANCE_H
#define ESSENTIA_VARIANCE_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Variance : public Algorithm {

 private:
  Input<std::vector<Real> > _array;
  Output<Real> _variance;

 public:
  Variance() {
    declareInput(_array, "array", "the input array");
    declareOutput(_variance, "variance", "the variance of the input array");
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

class Variance : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<Real> _variance;

 public:
  Variance() {
    declareAlgorithm("Variance");
    declareInput(_array, TOKEN, "array");
    declareOutput(_variance, TOKEN, "variance");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_VARIANCE_H
