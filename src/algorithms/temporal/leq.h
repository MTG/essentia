/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_LEQ_H
#define ESSENTIA_LEQ_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Leq : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<Real> _leq;

 public:
  Leq() {
    declareInput(_signal, "signal", "the input signal (must be non-empty)");
    declareOutput(_leq, "leq", "the equivalent sound level estimate");
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

class Leq : public AccumulatorAlgorithm {

 protected:
  Sink<Real> _signal;
  Source<Real> _leq;

  Real _energy;
  int _size;

 public:
  Leq() {
    declareInputStream(_signal, "signal", "the input signal (must be non-empty)");
    declareOutputResult(_leq, "leq", "the equivalent sound level estimate");
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


#endif // ESSENTIA_LEQ_H
