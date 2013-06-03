/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_GEOMETRICMEAN_H
#define ESSENTIA_GEOMETRICMEAN_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class GeometricMean : public Algorithm {

 private:
  Input<std::vector<Real> > _array;
  Output<Real> _geometricMean;

 public:
  GeometricMean() {
    declareInput(_array, "array", "the input array");
    declareOutput(_geometricMean, "geometricMean", "the geometric mean of the input array");
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

class GeometricMean : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<Real> _geometricMean;

 public:
  GeometricMean() {
    declareAlgorithm("GeometricMean");
    declareInput(_array, TOKEN, "array");
    declareOutput(_geometricMean, TOKEN, "geometricMean");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_GEOMETRICMEAN_H
