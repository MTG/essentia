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

#include "strongpeak.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* StrongPeak::name = "StrongPeak";
const char* StrongPeak::description = DOC("This algorithm extracts the Strong Peak from an audio spectrum. The Strong Peak is defined as the ratio between the spectrum's maximum peak's magnitude and the \"bandwidth\" of the peak above a threshold (half its amplitude). This ratio reveals whether the spectrum presents a very \"pronounced\" maximum peak (i.e. the thinner and the higher the maximum of the spectrum is, the higher the ratio value).\n\n"

"Note that \"bandwidth\" is defined as the width of the peak in the log10-frequency domain. This is different than as implemented in [1]. Using the log10-frequency domain allows this algorithm to compare strong peaks at lower frequencies with those from higher frequencies.\n\n"

"An exception is thrown if the input spectrum contains less than two elements.\n\n"

"References:\n"
"  [1] F. Gouyon and P. Herrera, \"Exploration of techniques for automatic\n"
"  labeling of audio drum tracks instruments,” in MOSART: Workshop on Current\n"
"  Directions in Computer Music, 2001.");


void StrongPeak::compute() {

  const std::vector<Real>& spectrum = _spectrum.get();
  Real& strongPeak = _strongPeak.get();

  if (spectrum.size() < 2) {
    throw EssentiaException("StrongPeak: the input spectrum size is less than 2 elements. StrongPeak ratio requires that a spectrum contains at least two elements");
  }

  int maxIndex = argmax(spectrum);
  int minIndex = argmin(spectrum);

  if (spectrum[minIndex] < 0) {
    throw EssentiaException("StrongPeak: input spectrum contains negative values");
  }

  Real maxMag = spectrum[maxIndex];

  if (maxMag == spectrum[minIndex]) {
    // flat spectrum
    strongPeak = 0;
    return;
  }

  Real threshold = maxMag / 2.0;

  // looking for the left bin of the bandwidth
  int bandwidthLeft = maxIndex;
  while (bandwidthLeft >= 0 && spectrum[bandwidthLeft] >= threshold) {
    bandwidthLeft--;
  }
  if (bandwidthLeft != 0) bandwidthLeft++;
  else if (spectrum[0] < threshold) {
    bandwidthLeft++;
  }
  // else bandwidthLeft should be zero because the 0th element is above the threshold

  // looking for the right bin of the bandwidth + 1
  int bandwidthRight = maxIndex;
  do {
    bandwidthRight++;
  } while (bandwidthRight < int(spectrum.size()) && spectrum[bandwidthRight] >= threshold);

  strongPeak = maxMag / log10( bandwidthRight / Real(bandwidthLeft) );
}
