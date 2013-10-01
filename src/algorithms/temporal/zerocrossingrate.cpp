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

#include "zerocrossingrate.h"
#include <cmath>

using namespace std;
using namespace essentia;
using namespace standard;


const char* ZeroCrossingRate::name = "ZeroCrossingRate";
const char* ZeroCrossingRate::description = DOC(
"This algorithm returns the zero-crossing rate of an audio signal. It is the number of sign changes between consecutive signal values divided by the total number of values. Noisy signals tend to have higher zero-crossing rate.\n"
"In order to avoid small variations around zero caused by noise, a threshold around zero is given to consider a valid zerocrosing whenever the boundary is crossed.\n"
"\n"
"Empty input signals will raise an exception.\n"
"\n"
"References:\n"
"  [1] Zero Crossing - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Zero-crossing_rate\n\n"
"  [2] G. Peeters, \"A large set of audio features for sound description\n"
"  (similarity and classification) in the CUIDADO project,\" CUIDADO I.S.T.\n"
"  Project Report, 2004");


void ZeroCrossingRate::configure() {
  _threshold = fabs(parameter("threshold").toReal());
}

void ZeroCrossingRate::compute() {

  const std::vector<Real>& signal = _signal.get();
  Real& zeroCrossingRate = _zeroCrossingRate.get();

  if (signal.empty()) throw EssentiaException("ZeroCrossingRate: the input signal is empty");

  zeroCrossingRate = 0.0;
  Real val = signal[0];
  if (std::fabs(val) < _threshold) val = 0;
  bool was_positive = (val > 0.0 );
  bool is_positive;

  for (int i=1; i<int(signal.size()); i++) {
    val = signal[i];
    if (std::fabs(val) <= _threshold) val = 0;
    is_positive = val > 0.0;
    if (was_positive != is_positive) {
      zeroCrossingRate++;
      was_positive = is_positive;
    }
  }

  zeroCrossingRate /= signal.size();
}

