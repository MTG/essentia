/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_ENERGY_H
#define ESSENTIA_ENERGY_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Energy : public Algorithm {

 private:
  Input<std::vector<Real> > _array;
  Output<Real> _energy;

 public:
  Energy() {
    declareInput(_array, "array", "the input array");
    declareOutput(_energy, "energy", "the energy of the input array");
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

class Energy : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<Real> _energy;

 public:
  Energy() {
    declareAlgorithm("Energy");
    declareInput(_array, TOKEN, "array");
    declareOutput(_energy, TOKEN, "energy");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_ENERGY_H
