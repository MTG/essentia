/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_TCTOTOTAL_H
#define ESSENTIA_TCTOTOTAL_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class TCToTotal : public Algorithm {

 protected:
  Input<std::vector<Real> > _envelope;
  Output<Real> _TCToTotal;

 public:
  TCToTotal() {
    declareInput(_envelope, "envelope", "the envelope of the signal (its length must be greater than 1");
    declareOutput(_TCToTotal, "TCToTotal", "the temporal centroid to total length ratio");
  }

  ~TCToTotal() {}

  void configure() {}
  void compute();
  void declareParameters() {}

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#include "accumulatoralgorithm.h"

namespace essentia {
namespace streaming {

class TCToTotal : public AccumulatorAlgorithm {

 protected:
  Sink<Real> _envelope;
  Source<Real> _TCToTotal;

  int _idx;
  double _num, _den;

 public:
  TCToTotal() {
    declareInputStream(_envelope, "envelope", "the envelope of the signal (its length must be greater than 1");
    declareOutputResult(_TCToTotal, "TCToTotal", "the temporal centroid to total length ratio");
    reset();
  }

  void declareParameters() {}

  void reset();

  void consume();
  void finalProduce();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_TCTOTOTAL_H
