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

#include "harmonicbpm.h"
#include "bpmutil.h"
#include <limits>

using namespace std;
using namespace essentia;
using namespace standard;

const char* HarmonicBpm::name = "HarmonicBpm";
const char* HarmonicBpm::category = "Rhythm";
const char* HarmonicBpm::description = DOC("This algorithm extracts bpms that are harmonically related to the tempo given by the \'bpm\' parameter.\n"
"The algorithm assumes a certain bpm is harmonically related to parameter bpm, when the greatest common divisor between both bpms is greater than threshold.\n"
"The \'tolerance\' parameter is needed in order to consider if two bpms are related. For instance, 120, 122 and 236 may be related or not depending on how much tolerance is given\n"
"\n"
"References:\n"
"  [1] Greatest common divisor - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Greatest_common_divisor");


void HarmonicBpm::configure() {
  _threshold = parameter("threshold").toReal();
  _bpm = parameter("bpm").toReal();
  _tolerance = parameter("tolerance").toReal();
}


vector<Real> HarmonicBpm::findHarmonicBpms(const vector<Real>& bpms) {
  // Accept only BPM values >= 1 for consistency with the `bpm` parameter.
  for(int i=0; i<int(bpms.size()); i++) {
    if (bpms[i] < 1)
      throw(EssentiaException("HarmonicBpm: bpm values below 1 are not allowed"));
  }

  vector<Real> harmonicBpms, harmonicRatios;
  harmonicBpms.reserve(bpms.size());
  harmonicRatios.reserve(bpms.size());

  Real mingcd = std::numeric_limits<int>::max();

  // Discard BPM values with GCD below the threshold.
  for (int i=0; i<int(bpms.size()); i++) {
     Real gcd = greatestCommonDivisor(_bpm, bpms[i], _tolerance);
     if (gcd >= _threshold) {
       harmonicBpms.push_back(bpms[i]);
       if (gcd < mingcd) mingcd = gcd;
     }
  }

  // Sort candidate BPMs in ascending order
  sort(harmonicBpms.begin(), harmonicBpms.end());

  vector<Real> bestHarmonicBpms;
  int i=0;
  Real prevBestBpm = -1;

  while (i<int(harmonicBpms.size())) {
    Real prevBpm = harmonicBpms[i];
    Real minError = std::numeric_limits<int>::max();
    Real bestBpm;

    // Select the best candidate among aproximately the same (within tolerance) BPMs.
    while (i < (int)harmonicBpms.size() &&
           areEqual(prevBpm,harmonicBpms[i], _tolerance)) {
      Real error=0, r=0;
      bpmDistance(_bpm, harmonicBpms[i], error, r);
      error = fabs(error);
      if (error < minError) {
        bestBpm = harmonicBpms[i];
        minError = error;
      }
      i++;
    }

    if (!areEqual(prevBestBpm, bestBpm, _tolerance)) {
      bestHarmonicBpms.push_back(bestBpm);
    }
    else {
      // If the selected candidate is too similar to the previous one (within
      // tolerance), keep the closest to the target BPM.
      Real e1=0, e2=0, r1=0, r2=0;
      bpmDistance(_bpm, bestHarmonicBpms[bestHarmonicBpms.size()-1], e1, r1);
      bpmDistance(_bpm, bestBpm, e2, r2);
      e1 = fabs(e1);
      e2 = fabs(e2);
      if (e1 > e2) {
        bestHarmonicBpms.back() = bestBpm;
      }
    }
    prevBestBpm = bestBpm;
  }

  return bestHarmonicBpms;
}


void HarmonicBpm::compute() {
  const vector<Real>& bpms = _bpmCandidates.get();
  vector<Real>& harmonicBpms = _harmonicBpms.get();
  harmonicBpms = findHarmonicBpms(bpms);
}

