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

#include "leq.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* Leq::name = "Leq";
const char* Leq::description = DOC("This algorithm computes the Equivalent sound level (Leq) of an audio signal. The Leq measure can be derived from the Revised Low-frequency B-weighting (RLB) or from the raw signal as described in [1]. If the signal contains no energy, Leq defaults to essentias definition of silence which is -90dB.\n"
"This algorithm will throw an exception on empty input.\n"
"\n"
"References:\n"
"  [1] G. A. Soulodre, \"Evaluation of Objective Loudness Meters,\" in\n"
"  The 116th AES Convention, 2004.");

void Leq::compute() {

  const std::vector<Real>& signal = _signal.get();
  Real& leq = _leq.get();

  if (signal.empty()) {
    throw EssentiaException("Leq: input signal is empty");
  }

  leq = pow2db(instantPower(signal));
}

} // namespace standard
} // namespace essentia


namespace essentia {
namespace streaming {

const char* Leq::name = standard::Leq::name;
const char* Leq::description = standard::Leq::description;

void Leq::reset() {
  AccumulatorAlgorithm::reset();
  _size = 0;
  _energy = 0;
}

void Leq::consume() {
  const vector<Real>& signal = *((const vector<Real>*)_signal.getTokens());

  _energy += energy(signal);
  _size += signal.size();
}

void Leq::finalProduce() {
  if (_size == 0) throw EssentiaException("Leq: signal is empty");

  _leq.push(pow2db(_energy/_size));
}


} // namespace streaming
} // namespace essentia
