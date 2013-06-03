/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_MEDIAN_H
#define ESSENTIA_MEDIAN_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Median : public Algorithm {

 private:
  Input<std::vector<Real> > _array;
  Output<Real> _median;

 public:
  Median() {
    declareInput(_array, "array", "the input array (must be non-empty)");
    declareOutput(_median, "median", "the median of the input array");
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

class Median : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<Real> _median;

 public:
  Median() {
    declareAlgorithm("Median");
    declareInput(_array, TOKEN, "array");
    declareOutput(_median, TOKEN, "median");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_MEDIAN_H
