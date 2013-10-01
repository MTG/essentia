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

#include "centralmoments.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* CentralMoments::name = "CentralMoments";
const char* CentralMoments::description = DOC("This algorithm extracts the 0th, 1st, 2nd, 3rd and 4th central moments of an array (i.e. it returns a 5-tuple in which the index corresponds to the order of the moment).\n"
"Note:\n"
" - For a spectral centroid, frequency range should be equal to samplerate/2\n"
" - For an audio centroid, frequency range should be equal to (audio_size-1) / samplerate\n"
"\n"
"Central moments cannot be computed on arrays which size is less than 2, in which case an exception is thrown.\n"
"\n"
"References:\n"
"  [1] Sample Central Moment -- from Wolfram MathWorld,\n"
"  http://mathworld.wolfram.com/SampleCentralMoment.html\n\n"
"  [2] Central Moment - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Central_moment");

void CentralMoments::configure() {
  _range = parameter("range").toReal();
}

void CentralMoments::compute() {

  // For precision reasons, we first compute the central moments with a normalized
  // range [0,1], and we multiply by the desired range at the end only.

  const std::vector<Real>& array = _array.get();
  std::vector<Real>& centralMoments = _centralMoments.get();
  centralMoments.resize(5);

  if (array.empty()) {
    throw EssentiaException("CentralMoments: cannot compute the central moments of an empty array");
  }

  if (array.size() == 1) {
    throw EssentiaException("CentralMoments: cannot compute the central moments of an array of size 1");
  }

  int arraySize = array.size();

  // scale is the horizontal scale, thus i*scale corresponds to the
  // normalized frequency, i.e.: between 0 and 1
  double scale = (double)1.0 / (arraySize - 1);

  double norm = 0.0;
  for (int i=0; i<arraySize; i++) norm += array[i];

  if (norm == 0.0) {
    for (int k=0; k<5; k++) {
      centralMoments[k] = 0.0;
    }
    return;
  }

  // centroid is also in normalized frequency, i.e.: between 0 and 1
  double centroid = 0.0;
  for (int i=0; i<arraySize; i++) {
    centroid += (i*scale) * array[i];
  }
  centroid /= norm;

  centralMoments[0] = 1.0;
  centralMoments[1] = 0.0;

  double m2 = 0.0, m3 = 0.0, m4 = 0.0;
  double v, v2, v2f;

  for (int i=0; i<arraySize; i++) {
    v = (i*scale) - centroid;
    v2 = v*v;
    v2f = v2 * array[i];
    m2 += v2f;
    m3 += v2f * v;
    m4 += v2f * v2;
  }

  m2 /= norm;
  m3 /= norm;
  m4 /= norm;

  // we want the results inside the specified range, so as we factored it out
  // in the above formula, we have to inject it again to get back the results
  // relative to the desired range.
  double r = _range;
  centralMoments[2] = m2 * r*r;
  centralMoments[3] = m3 * r*r*r;
  centralMoments[4] = m4 * r*r*r*r;

}
