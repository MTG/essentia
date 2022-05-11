/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

#include "tctototal.h"

using namespace std;

namespace essentia {
namespace standard {

const char* TCToTotal::name = "TCToTotal";
const char* TCToTotal::category = "Envelope/SFX";
const char* TCToTotal::description = DOC("This algorithm calculates the ratio of the temporal centroid to the total length of a signal envelope. This ratio shows how the sound is 'balanced'. Its value is close to 0 if most of the energy lies at the beginning of the sound (e.g. decrescendo or impulsive sounds), close to 0.5 if the sound is symetric (e.g. 'delta unvarying' sounds), and close to 1 if most of the energy lies at the end of the sound (e.g. crescendo sounds).\n"
"\n"
"Please note that the TCToTotal ratio will return 0.5 for a zero signal (a signal consisting of only zeros) as 0.5 is the middle point of the signal. TCToTotal is not defined for a signal of less than 2 elements."
"An exception is thrown if the given envelope's size is not larger than 1.\n\n"
"This algorithm is intended to be plugged after the Envelope algorithm");

void TCToTotal::compute() {

  const vector<Real>& envelope = _envelope.get();
  Real& TCToTotal = _TCToTotal.get();

  if (envelope.size() < 2) {
    throw EssentiaException("TCToTotal: the given envelope's size is not larger than 1");
  }

  double num = 0.0;
  double den = 0.0;
  for (int i=0; i<int(envelope.size()); i++) {
    num += envelope[i] * i;
    den += envelope[i];
  }

  if (den == 0) {
    // Singal envelope consists of 0 or integral of singal is 0, then set normalizez temporal centroid to 0.5 (middle position)
    TCToTotal = 0.5;
  } else {
    double centroid = num / den;
    TCToTotal = centroid / double(envelope.size()-1);
  }
}

} // namespace standard
} // namespace essentia


namespace essentia {
namespace streaming {

const char* TCToTotal::name = essentia::standard::TCToTotal::name;
const char* TCToTotal::description = essentia::standard::TCToTotal::description;

void TCToTotal::consume() {
  const vector<Real>& envelope = *((const vector<Real>*)_envelope.getTokens());

  for (int i=0; i<(int)envelope.size(); i++) {
    _num += envelope[i] * _idx;
    _den += envelope[i];
    _idx++;
  }
}

void TCToTotal::finalProduce() {
  
  if (_idx < 2) {
    throw EssentiaException("TCToTotal: the given envelope is not larger than 1 element");
  }
  
  if (_den == 0) {
    // Singal envelope consists of 0 or integral of singal is 0, then set normalizez temporal centroid to 0.5 (middle position)
    _TCToTotal.push(Real(0.5));
    return;
  }

  double centroid = _num / _den;

  // _idx also represents the total number of tokens consumed
  _TCToTotal.push(Real(centroid / double(_idx-1)));
}

void TCToTotal::reset() {
  AccumulatorAlgorithm::reset();
  _idx = 0;
  _num = _den = 0.0;
}

} // namespace streaming
} // namespace essentia
