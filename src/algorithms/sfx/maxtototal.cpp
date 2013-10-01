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

#include "maxtototal.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* MaxToTotal::name = "MaxToTotal";
const char* MaxToTotal::description = DOC("This algorithm computes the ratio between the index of the maximum value of the envelope of a signal and the total length of the envelope. This ratio shows how much the maximum amplitude is off-center. Its value is close to 0 if the maximum is close to the beginning (e.g. Decrescendo or Impulsive sounds), close to 0.5 if it is close to the middle (e.g. Delta sounds) and close to 1 if it is close to the end of the sound (e.g. Crescendo sounds). This algorithm is intended to be fed by the output of the Envelope algorithm\n\n"
"MaxToTotal will throw an exception if the input envelope is empty.");

void MaxToTotal::compute() {

  const vector<Real>& envelope = _envelope.get();
  Real& maxToTotal = _maxToTotal.get();

  if (envelope.empty()) {
    throw EssentiaException("MaxToTotal: envelope is empty, maxToTotal is not defined for an empty envelope");
  }

  maxToTotal = Real(argmax(envelope)) / envelope.size();
}

} // namespace standard
} // namespace essentia


namespace essentia {
namespace streaming {

const char* MaxToTotal::name = essentia::standard::MaxToTotal::name;
const char* MaxToTotal::description = essentia::standard::MaxToTotal::description;


void MaxToTotal::consume() {
  const vector<Real>& envelope = *((const vector<Real>*)_envelope.getTokens());

  int maxIdx = argmax(envelope);
  if (envelope[maxIdx] > _max) {
    _max = envelope[maxIdx];
    _maxIdx = maxIdx + _size;
  }

  _size += envelope.size();
}

void MaxToTotal::finalProduce() {
  if (_size == 0) {
    throw EssentiaException("MaxToTotal: envelope is empty, maxToTotal is not defined for an empty envelope");
  }

  _maxToTotal.push(Real(_maxIdx) / _size);
}

void MaxToTotal::reset() {
  AccumulatorAlgorithm::reset();
  _size = 0;
  _max = std::numeric_limits<Real>::min();
  _maxIdx = 0;
}

} // namespace streaming
} // namespace essentia
