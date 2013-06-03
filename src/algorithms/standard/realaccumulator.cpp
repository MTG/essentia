/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "realaccumulator.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* RealAccumulator::name = "RealAccumulator";
const char* RealAccumulator::description = DOC("This algorithm takes a stream of Real values "
"and outputs them as a single vector when the end of the stream is reached.");


RealAccumulator::RealAccumulator() {
  declareInput(_value, "data", "the input signal");
  declareOutput(_array, 0, "array", "the accumulated signal in one single frame");

  _vectorOutput = new VectorOutput<Real>(&_accu);
  _value >> _vectorOutput->input("data");
}


RealAccumulator::~RealAccumulator() {
  delete _vectorOutput;
}


void RealAccumulator::reset() {
  AlgorithmComposite::reset();
  _accu.clear();
}


AlgorithmStatus RealAccumulator::process() {
  if (!shouldStop()) return PASS;

  _array.push(_accu);
  return FINISHED;
}

} // namespace streaming
} // namespace essentia
