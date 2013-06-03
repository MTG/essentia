/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "derivative.h"

using namespace std;

namespace essentia {
namespace standard {

const char* Derivative::name = "Derivative";
const char* Derivative::description = DOC("This algorithm returns the first-order derivative of the input signal, ie: for each input value, it returns the value minus the previous one.");

void Derivative::compute() {
  const std::vector<Real>& input = _input.get();
  std::vector<Real>& output = _output.get();
  int size = input.size();
  output.resize(size);

  output[0] = input[0];
  for (int i=1; i<size; ++i) {
    output[i] = input[i] - input[i-1];
  }
}

} // namespace standard
} // namespace essentia


namespace essentia {
namespace streaming {

const char* Derivative::name = standard::Derivative::name;
const char* Derivative::description = standard::Derivative::description;

void Derivative::reset() {
  Algorithm::reset();
  _oldValue = 0;
}

void Derivative::configure() {
  reset();
}

AlgorithmStatus Derivative::process() {
  AlgorithmStatus status = acquireData();

  if (status != OK) return status;

  const Real& input = _input.firstToken();
  Real& output = _output.firstToken();

  output = input - _oldValue;
  _oldValue = input;

  releaseData();

  return OK;
}

} // namespace streaming
} // namespace essentia
