/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_BINARYOPERATOR_H
#define ESSENTIA_BINARYOPERATOR_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class BinaryOperator : public Algorithm {

 private:
  Input<std::vector<Real> > _input1;
  Input<std::vector<Real> > _input2;
  Output<std::vector<Real> > _output;

 public:
  BinaryOperator() {
    declareInput(_input1, "input1", "the left-hand side input");
    declareInput(_input2, "input2", "the right-hand side input");
    declareOutput(_output, "output", "the resulting vector");
  }

  void declareParameters() {
    declareParameter("type", "the type of the operation", "{+,-,*,/}", "+");
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

class BinaryOperator : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _input1;
  Sink<std::vector<Real> > _input2;
  Source<std::vector<Real> > _output;

 public:
  BinaryOperator() {
    declareAlgorithm("BinaryOperator");
    declareInput(_input1, TOKEN, "input1");
    declareInput(_input2, TOKEN, "input2");
    declareOutput(_output, TOKEN, "output");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_BINARYOPERATOR_H
