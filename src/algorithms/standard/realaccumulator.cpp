/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of Essentia
 *
 * Essentia is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
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
