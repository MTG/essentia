/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "flatnesssfx.h"
#include <algorithm>

using namespace std;
using namespace essentia;
using namespace standard;

const char* FlatnessSFX::name = "FlatnessSFX";
const char* FlatnessSFX::description = DOC(
  "This algorithm calculates the flatness coefficient of a signal envelope.\n"
  "There are two thresholds defined: a lower one at 20% and an upper one at 95%. "
  "The thresholds yield two values: one value which has 20% of the total values underneath, "
  "and one value which has 95% of the total values underneath. "
  "The flatness coefficient is then calculated as the ratio of these two values. "
  "This algorithm is meant to be plugged after Envelope algorithm, however in streaming mode "
  "a RealAccumulator algorithm should be connected in between the two.\n"
  "In the current form the algorithm can't be calculated in streaming mode, "
  "since it would violate the streaming mode policy of having low memory consumption.\n"
  "An exception is thrown if the input envelope is empty."
);

const Real FlatnessSFX::lowerThreshold = 20.0;
const Real FlatnessSFX::upperThreshold = 95.0;

// envelope needs to be sorted
Real FlatnessSFX::rollOff(const vector<Real>& sortedEnvelope, Real threshold) const {
  if (threshold < 0 || threshold > 100.0) {
    throw EssentiaException("FlatnessSFX: threshold out of bounds");
  }

  int max_index = sortedEnvelope.size() - 1;
  Real indexf = (Real)max_index * threshold / 100.0;

  if (indexf == (int)indexf) {
    return sortedEnvelope[(int)indexf];
  }
  else {
    int index = int(indexf);
    return (sortedEnvelope[index + 1] - sortedEnvelope[index]) * (indexf - index) + sortedEnvelope[index];
  }
}

void FlatnessSFX::compute() {
  const vector<Real>& envelope = _envelope.get();
  Real& flatnessSFX = _flatnessSFX.get();

  if (envelope.empty()) {
    throw EssentiaException("FlatnessSFX: input signal is empty");
  }

  vector<Real> sortedEnvelope = envelope;
  sort(sortedEnvelope.begin(), sortedEnvelope.end());

  Real num = rollOff(sortedEnvelope, upperThreshold);
  Real den = rollOff(sortedEnvelope, lowerThreshold);
  if (den == 0.0) {
    flatnessSFX = 1.0;
  }
  else {
    flatnessSFX = num / den;
  }
}
