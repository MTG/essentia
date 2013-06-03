/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_POWERMEAN_H
#define ESSENTIA_POWERMEAN_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class PowerMean : public Algorithm {

 private:
  Input<std::vector<Real> > _array;
  Output<Real> _powerMean;
  Algorithm* _geometricMean;

 public:
  PowerMean() {
    declareInput(_array, "array", "the input array (must contain only positive real numbers)");
    declareOutput(_powerMean, "powerMean", "the power mean of the input array");

    _geometricMean = AlgorithmFactory::create("GeometricMean");
  }

  ~PowerMean() {
    delete _geometricMean;
  }

  void declareParameters() {
    declareParameter("power", "the power to which to elevate each element before taking the mean", "(-inf,inf)", 1.0);
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

class PowerMean : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<Real> _powerMean;

 public:
  PowerMean() {
    declareAlgorithm("PowerMean");
    declareInput(_array, TOKEN, "array");
    declareOutput(_powerMean, TOKEN, "powerMean");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_POWERMEAN_H
