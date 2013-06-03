/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "effectiveduration.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* EffectiveDuration::name = "EffectiveDuration";
const char* EffectiveDuration::description = DOC(
"This algorithm returns the effective duration of an envelope signal. The effective duration is a measure of the time the signal is perceptually meaningful. This is approximated by the time the envelope is above or equal to a given threshold. This measure allows to distinguish percussive sounds from sustained sounds but depends on the signal length.\n"
"This algorithm uses 40% of the envelope maximum as the threshold.\n"
"References:\n"
"  [1] G. Peeters, A large set of audio features for sound description (similarity and classification) in the CUIDADO project,"
"      CUIDADO I.S.T. Project Report, 2004");

const Real EffectiveDuration::thresholdRatio = 0.4;
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
  Real threshold = thresholdRatio * maxValue;
  if (threshold < noiseFloor) threshold = noiseFloor;

  for (int i=0; i<int(signal.size()); i++) {
    if (fabs(signal[i]) >= threshold) nSamplesAboveThreshold++;
  }

  effectiveDuration = (Real)nSamplesAboveThreshold / parameter("sampleRate").toReal();
}
