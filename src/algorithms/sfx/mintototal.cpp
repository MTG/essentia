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

#include "mintototal.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* MinToTotal::name = "MinToTotal";
const char* MinToTotal::description = DOC("This algorithm computes the ratio between the index of the minimum value of the envelope of a signal and the total length of the envelope.\n\n"
"An exception is thrown if the input envelop is empty.");

void MinToTotal::compute() {

  const vector<Real>& envelope = _envelope.get();
  Real& minToTotal = _minToTotal.get();

  if (envelope.empty()) {
    throw EssentiaException("MinToTotal: envelope is empty, minToTotal is not defined for an empty envelope");
  }

  minToTotal = Real(argmin(envelope)) / envelope.size();
}

} // namespace standard
} // namespace essentia


namespace essentia {
namespace streaming {

const char* MinToTotal::name = essentia::standard::MinToTotal::name;
const char* MinToTotal::description = essentia::standard::MinToTotal::description;


void MinToTotal::consume() {
  const vector<Real>& envelope = *((const vector<Real>*)_envelope.getTokens());

  int minIdx = argmin(envelope);
  if (envelope[minIdx] < _min) {
    _min = envelope[minIdx];
    _minIdx = minIdx + _size;
  }

  _size += envelope.size();
}

void MinToTotal::finalProduce() {
  if (_size == 0) {
    throw EssentiaException("MinToTotal: envelope is empty, minToTotal is not defined for an empty envelope");
  }

  _minToTotal.push(Real(_minIdx) / _size);
}

void MinToTotal::reset() {
  AccumulatorAlgorithm::reset();
  _size = 0;
  _min = std::numeric_limits<Real>::max();
  _minIdx = 0;
}

} // namespace streaming
} // namespace essentia
