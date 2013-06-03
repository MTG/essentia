/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_POLAR2CARTESIAN_H
#define ESSENTIA_POLAR2CARTESIAN_H

#include "algorithm.h"
#include "streamingalgorithmwrapper.h"
#include <complex>

namespace essentia {
namespace standard {

class PolarToCartesian : public Algorithm {

 private:
  Input<std::vector<Real> > _magnitude;
  Input<std::vector<Real> > _phase;
  Output<std::vector<std::complex<Real> > > _complex;

 public:
  PolarToCartesian() {
    declareInput(_magnitude, "magnitude", "the magnitude vector");
    declareInput(_phase, "phase", "the phase vector");
    declareOutput(_complex, "complex", "the resulting complex vector");
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
namespace streaming {

class PolarToCartesian : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _magnitude;
  Sink<std::vector<Real> > _phase;
  Source<std::vector<std::complex<Real> > > _complex;

 public:
  PolarToCartesian() {
    declareAlgorithm("PolarToCartesian");
    declareInput(_magnitude, TOKEN, "magnitude");
    declareInput(_phase, TOKEN, "phase");
    declareOutput(_complex, TOKEN, "complex");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_POLAR2CARTESIAN_H
