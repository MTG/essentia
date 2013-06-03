/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */


#ifndef ESSENTIA_MAGNITUDE_H
#define ESSENTIA_MAGNITUDE_H


#include "algorithm.h"
#include "streamingalgorithmwrapper.h"
#include <complex>


namespace essentia {
namespace standard {


class Magnitude : public Algorithm {

 private:
  Input<std::vector<std::complex<Real> > > _complex;
  Output<std::vector<Real> > _magnitude;

 public:
  Magnitude() {
    declareInput(_complex, "complex", "the input vector of complex numbers");
    declareOutput(_magnitude, "magnitude", "the magnitudes of the input vector");
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;
};


} // namespace standard
namespace streaming {


class Magnitude : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<std::complex<Real> > > _complex;
  Source<std::vector<Real> > _magnitude;

 public:
  Magnitude() {
    declareAlgorithm("Magnitude");
    declareInput(_complex, TOKEN, "complex");
    declareOutput(_magnitude, TOKEN, "magnitude");
  }
};


} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_MAGNITUDE_H
