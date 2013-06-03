/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_MAXTOTOTAL_H
#define ESSENTIA_MAXTOTOTAL_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class MaxToTotal : public Algorithm {

 protected:
  Input<std::vector<Real> > _envelope;
  Output<Real> _maxToTotal;

 public:
  MaxToTotal() {
    declareInput(_envelope, "envelope", "the envelope of the signal");
    declareOutput(_maxToTotal, "maxToTotal", "the maximum amplitude position to total length ratio");
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;
};

} // namespace essentia
} // namespace standard

#include "accumulatoralgorithm.h"

namespace essentia {
namespace streaming {

class MaxToTotal : public AccumulatorAlgorithm {

 protected:
  Sink<Real> _envelope;
  Source<Real> _maxToTotal;

  int _size, _maxIdx;
  Real _max;

 public:
  MaxToTotal() {
    declareInputStream(_envelope, "envelope", "the envelope of the signal");
    declareOutputResult(_maxToTotal, "maxToTotal", "the maximum amplitude position to total length ratio");
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

#endif // ESSENTIA_MAXTOTOTAL_H
