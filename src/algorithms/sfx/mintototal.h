/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_MINTOTOTAL_H
#define ESSENTIA_MINTOTOTAL_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class MinToTotal : public Algorithm {

 protected:
  Input<std::vector<Real> > _envelope;
  Output<Real> _minToTotal;

 public:
  MinToTotal() {
    declareInput(_envelope, "envelope", "the envelope of the signal");
    declareOutput(_minToTotal, "minToTotal", "the minimum amplitude position to total length ratio");
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "accumulatoralgorithm.h"

namespace essentia {
namespace streaming {

class MinToTotal : public AccumulatorAlgorithm {

 protected:
  Sink<Real> _envelope;
  Source<Real> _minToTotal;

  int _size, _minIdx;
  Real _min;

 public:
  MinToTotal() {
    declareInputStream(_envelope, "envelope", "the envelope of the signal");
    declareOutputResult(_minToTotal, "minToTotal", "the minimum amplitude position to total length ratio");
    reset();
  }

  void reset();
  void declareParameters() {}

  void consume();
  void finalProduce();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_MINTOTOTAL_H
