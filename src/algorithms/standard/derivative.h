/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef DERIVATIVE_H
#define DERIVATIVE_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Derivative : public Algorithm {

 private:
  Input<std::vector<Real> > _input;
  Output<std::vector<Real> > _output;

 public:
  Derivative() {
    declareInput(_input, "signal", "the input signal");
    declareOutput(_output, "signal", "the derivative of the input signal");
  }
  ~Derivative() {}
  void declareParameters() {}
  void compute();
  void configure() {}

  static const char* name;
  static const char* description;
};

}// namespace standard
}// namespace essentia

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {


class Derivative : public Algorithm {

 protected:
  Sink<Real> _input;
  Source<Real> _output;
  Real _oldValue;

 public:
  Derivative() {
    declareInput(_input, 1, "signal", "the input signal");
    declareOutput(_output, 1, "signal", "the derivative of the input signal");
  }

  ~Derivative() {}

  void reset();
  void declareParameters() {}
  void configure();
  AlgorithmStatus process();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // DERIVATIVE_H
