/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_MEAN_H
#define ESSENTIA_MEAN_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Mean : public Algorithm {

 private:
  Input<std::vector<Real> > _array;
  Output<Real> _mean;

 public:
  Mean() {
    declareInput(_array, "array", "the input array");
    declareOutput(_mean, "mean", "the mean of the input array");
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

class Mean : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<Real> _mean;

 public:
  Mean() {
    declareAlgorithm("Mean");
    declareInput(_array, TOKEN, "array");
    declareOutput(_mean, TOKEN, "mean");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_MEAN_H
