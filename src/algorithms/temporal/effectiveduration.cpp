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

#include "effectiveduration.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* EffectiveDuration::name = "EffectiveDuration";
const char* EffectiveDuration::category = "Duration/silence";
const char* EffectiveDuration::description = DOC(
"This algorithm computes the effective duration of an envelope signal. The effective duration is a measure of the time the signal is perceptually meaningful. This is approximated by the time the envelope is above or equal to a given threshold and is above the -90db noise floor. This measure allows to distinguish percussive sounds from sustained sounds but depends on the signal length.\n"
"By default, this algorithm uses 40% of the envelope maximum as the threshold which is suited for short sounds. Note, that the 0% thresold corresponds to the duration of signal above -90db noise floor, while the 100% thresold corresponds to the number of times the envelope takes its maximum value.\n"
"References:\n"
"  [1] G. Peeters, \"A large set of audio features for sound description\n"
"  (similarity and classification) in the CUIDADO project,\" CUIDADO I.S.T.\n"
"  Project Report, 2004");

const Real EffectiveDuration::noiseFloor = db2amp(-90); // -90db is silence (see essentiamath.h)

void EffectiveDuration::compute() {

  const std::vector<Real>& signal = _signal.get();
  Real& effectiveDuration = _effectiveDuration.get();

  // calculate max amplitude
  Real maxValue = 0; // always positive
  for (int i=0; i<int(signal.size()); ++i) {
    if (fabs(signal[i]) > maxValue) maxValue = fabs(signal[i]);
  }

  // count how many samples are above max amplitude
  int nSamplesAboveThreshold = 0;
  Real threshold = parameter("thresholdRatio").toReal() * maxValue;
  if (threshold < noiseFloor) threshold = noiseFloor;

  for (int i=0; i<int(signal.size()); i++) {
    if (fabs(signal[i]) >= threshold) nSamplesAboveThreshold++;
  }

  effectiveDuration = (Real)nSamplesAboveThreshold / parameter("sampleRate").toReal();
}
