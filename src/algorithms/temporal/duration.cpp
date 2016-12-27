/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#include "duration.h"
using namespace std;

namespace essentia {
namespace standard {

const char* Duration::name = "Duration";
const char* Duration::category = "Duration/silence";
const char* Duration::description = DOC("This algorithm outputs the total duration of an audio signal.");

void Duration::compute() {
  const vector<Real>& signal = _signal.get();
  Real& duration = _duration.get();

  duration = signal.size()/parameter("sampleRate").toReal();
}


} // namespace standard
} // namespace streaming


namespace essentia {
namespace streaming {

const char* Duration::name = essentia::standard::Duration::name;
const char* Duration::category = essentia::standard::Duration::category;
const char* Duration::description = essentia::standard::Duration::description;

void Duration::reset() {
  AccumulatorAlgorithm::reset();
  _nsamples = 0;
}

void Duration::consume() {
  const vector<Real>& signal = *((const vector<Real>*)_signal.getTokens());

  _nsamples += signal.size();
}

void Duration::finalProduce() {
  _duration.push((Real)(_nsamples / parameter("sampleRate").toReal()));
}

} // namespace streaming
} // namespace essentia
