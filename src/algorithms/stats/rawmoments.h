/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_RAWMOMENTS_H
#define ESSENTIA_RAWMOMENTS_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class RawMoments : public Algorithm {

 private:
  Input<std::vector<Real> > _array;
  Output<std::vector<Real> > _rawMoments;

 public:
  RawMoments() {
    declareInput(_array, "array", "the input array");
    declareOutput(_rawMoments, "rawMoments", "the (raw) moments of the input array");
  }

  void declareParameters() {
    declareParameter("range", "the range of the input array, used for normalizing the results", "(0,inf)", 22050.);
  }

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class RawMoments : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<std::vector<Real> > _rawMoments;

 public:
  RawMoments() {
    declareAlgorithm("RawMoments");
    declareInput(_array, TOKEN, "array");
    declareOutput(_rawMoments, TOKEN, "rawMoments");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_RAWMOMENTS_H
