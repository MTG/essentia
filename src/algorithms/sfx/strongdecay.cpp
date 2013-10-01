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

#include "strongdecay.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* StrongDecay::name = "StrongDecay";
const char* StrongDecay::description = DOC("This algorithm extracts the Strong Decay of an audio signal. The Strong Decay is built from the non-linear combination of the signal energy and the signal temporal centroid, the latter being the balance of the absolute value of the signal. A signal containing a temporal centroid near its start boundary and a strong energy is said to have a strong decay.\n"
"\n"
"This algorithm is not defined for zero signals (i.e. silence) nor when the signal's size is less than two, as it could not compute its centroid.\n"
"\n"
"References:\n"
"  [1] F. Gouyon and P. Herrera, \"Exploration of techniques for automatic\n"
"  labeling of audio drum tracks instruments,\" in MOSART: Workshop on Current\n"
"  Directions in Computer Music, 2001.");

void StrongDecay::compute() {

  const vector<Real>& signal = _signal.get();
  Real& strongDecay = _strongDecay.get();

  vector<Real> absSignal;
  Real centroid;

  _abs->input("array").set(signal);
  _abs->output("array").set(absSignal);
  _abs->compute();
  _centroid->configure("range", (signal.size()-1) / parameter("sampleRate").toReal());
  _centroid->input("array").set(absSignal);
  _centroid->output("centroid").set(centroid);
  _centroid->compute();

  if (centroid <= 0.0) {
    throw EssentiaException("StrongDecay: the strong decay is not defined for a zero signal");
  }

  Real signalEnergy = energy(signal);
  strongDecay = sqrt(signalEnergy / centroid);
}

} // namespace standard
} // namespace essentia



namespace essentia {
namespace streaming {

const char* StrongDecay::name = standard::StrongDecay::name;
const char* StrongDecay::description = standard::StrongDecay::description;


void StrongDecay::reset() {
  AccumulatorAlgorithm::reset();
  _centroid = 0.0;
  _energy = 0.0;
  _weights = 0.0;
  _idx = 0;
}

void StrongDecay::consume() {
  const vector<Real>& signal = *((const vector<Real>*)_signal.getTokens());

  for (int i=0; i<(int)signal.size(); i++) {
    Real absSignal = fabs(signal[i]);
    _centroid += (_idx++)*absSignal;
    _weights += absSignal;
  }
  _energy += energy(signal);
}


void StrongDecay::finalProduce() {
  if (_idx < 2) {
    throw EssentiaException("StrongDecay: cannot compute centroid of an array of size < 2");
  }

  if (_weights == 0) {
    _centroid = 0.0;
  }
  else {
    _centroid /= _weights;
    _centroid /= parameter("sampleRate").toReal();
  }

  if (_centroid <= 0.0) {
    throw EssentiaException("StrongDecay: the strong decay is not defined for a zero signal");
  }

  _strongDecay.push((Real)sqrt(_energy/_centroid));
}

} // namespace streaming
} // namespace essentia
