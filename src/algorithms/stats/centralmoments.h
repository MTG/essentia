/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_CENTRALMOMENTS_H
#define ESSENTIA_CENTRALMOMENTS_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class CentralMoments : public Algorithm {

 private:
  Input<std::vector<Real> > _array;
  Output<std::vector<Real> > _centralMoments;

 protected:
  Real _range;

 public:
  CentralMoments() {
    declareInput(_array, "array", "the input array");
    declareOutput(_centralMoments, "centralMoments", "the central moments of the input array");
  }

  void declareParameters() {
    declareParameter("range", "the range of the input array, used for normalizing the results", "(0,inf)", 1.0);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class CentralMoments : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<std::vector<Real> > _centralMoments;

 public:
  CentralMoments() {
    declareAlgorithm("CentralMoments");
    declareInput(_array, TOKEN, "array");
    declareOutput(_centralMoments, TOKEN, "centralMoments");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_CENTRALMOMENTS_H
