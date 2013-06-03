/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_CARTESIANTOPOLAR_H
#define ESSENTIA_CARTESIANTOPOLAR_H


#include "algorithm.h"
#include "streamingalgorithmwrapper.h"
#include <complex>


namespace essentia {
namespace standard {


class CartesianToPolar : public Algorithm {

 private:
  Input<std::vector<std::complex<Real> > > _complex;
  Output<std::vector<Real> > _magnitude;
  Output<std::vector<Real> > _phase;

 public:
  CartesianToPolar() {
    declareInput(_complex, "complex", "the complex input vector");
    declareOutput(_magnitude, "magnitude", "the magnitude vector");
    declareOutput(_phase, "phase", "the phase vector");
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;
};


} // namespace standard
namespace streaming {


class CartesianToPolar : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<std::complex<Real> > > _complex;
  Source<std::vector<Real> > _magnitude;
  Source<std::vector<Real> > _phase;

 public:
  CartesianToPolar() {
    declareAlgorithm("CartesianToPolar");
    declareInput(_complex, TOKEN, "complex");
    declareOutput(_magnitude, TOKEN, "magnitude");
    declareOutput(_phase, TOKEN, "phase");
  }
};


} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_CARTESIANTOPOLAR_H
