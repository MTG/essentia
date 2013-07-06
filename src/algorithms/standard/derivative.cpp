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
