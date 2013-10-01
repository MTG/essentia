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

#include "rawmoments.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* RawMoments::name = "RawMoments";
const char* RawMoments::description = DOC("This algorithm computes the first 5 raw moments of an array of Reals. The output array is of size 6 because the zero-ith moment is used for padding so that the first moment corresponds to index 1.\n\n"

"Note:\n"
"  This algorithm has a range parameter, which usually represents a frequency (results will range from 0 to range). For a spectral centroid, the range should be equal to samplerate / 2. For an audio centroid, the frequency range should be equal to (audio_size-1) / samplerate.\n\n"

"An exception is thrown if the input array's size is smaller than 2.\n\n"

"References:\n"
"  [1] Raw Moment -- from Wolfram MathWorld,\n"  
"  http://mathworld.wolfram.com/RawMoment.html");

void RawMoments::compute() {

  /*
  The algorithm implemented here is:

  define i_in_freq = (i * samplerate/2) / (framesize-1)

  p[i] = spectrum[i] / sum(spectrum[i], i=0..size-1)

  mu = sum(i_in_freq * p[i], i=0..size-1)

  raw_moment[order] = sum((i_in_freq - mu)^order * p[i], i=0..size-1)
  */

  const std::vector<Real>& frame = _array.get();
  std::vector<Real>& rawMoments = _rawMoments.get();
  rawMoments.resize(5);

  if (frame.size() <= 1) {
    throw EssentiaException("RawMoments: the input array size is smaller than 2");
  }

  int frameSize = frame.size() - 1;
  // scale is the horizontal scale, thus i*scale corresponds to the
  // normalized frequency, i.e.: between 0 and 1
  Real scale = 1.0 / frameSize;

  double norm = 0.0;
  for (int i=0; i<int(frame.size()); i++) norm += frame[i];

  if (norm == 0.0) {
    for (int k=0; k<5; k++) {
      rawMoments[k] = 0.0;
    }
    return;
  }

  // centroid is also in normalized frequency, i.e.: between 0 and 1
  Real centroid = 0.0;
  for (int i=0; i<int(frame.size()); i++) {
    centroid += (i*scale) * frame[i];
  }
  centroid /= norm;

  rawMoments[0] = 1.0;
  rawMoments[1] = centroid * parameter("range").toReal();

  for (int k=2; k<5; k++) {
    Real tmp = 0.0;

    for (int i=0; i<int(frame.size()); i++) {
      tmp += frame[i] * pow(i*scale, (int)k);
    }

    tmp /= norm;

    // we want the results in Hz, not in normalized frequency, so as we
    // factored out samplingRate in the above formula, we have to inject
    // it again to get back the results relative to the frequency range.
    rawMoments[k] = tmp * pow(parameter("range").toReal(), k); // renormalize to frequency range
  }
}
