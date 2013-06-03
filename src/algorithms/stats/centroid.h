/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_CENTROID_H
#define ESSENTIA_CENTROID_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Centroid : public Algorithm {

 protected:
  Input<std::vector<Real> > _array;
  Output<Real> _centroid;

  Real _range;

 public:
  Centroid() {
    declareInput(_array, "array", "the input array");
    declareOutput(_centroid, "centroid", "the centroid of the array");
  }

  void declareParameters() {
    declareParameter("range", "the range of the input array, used for normalizing the results", "(0,inf)", 1.0);
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

class Centroid : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<Real> _centroid;

 public:
  Centroid() {
    declareAlgorithm("Centroid");
    declareInput(_array, TOKEN, "array");
    declareOutput(_centroid, TOKEN, "centroid");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_CENTROID_H
