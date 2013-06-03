/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_DECREASE_H
#define ESSENTIA_DECREASE_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Decrease : public Algorithm {

 private:
  Input<std::vector<Real> > _array;
  Output<Real> _decrease;

 public:
  Decrease() {
    declareInput(_array, "array", "the input array");
    declareOutput(_decrease, "decrease", "the decrease of the input array");
  }

  void declareParameters() {
    declareParameter("range", "the range of the input array, used for normalizing the results", "(-inf,inf)", 1.0);
  }

  void configure();

  void compute();

  static const char* name;
  static const char* description;

 protected:
  Real _range;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Decrease : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<Real> _decrease;

 public:
  Decrease() {
    declareAlgorithm("Decrease");
    declareInput(_array, TOKEN, "array");
    declareOutput(_decrease, TOKEN, "decrease");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_DECREASE_H
