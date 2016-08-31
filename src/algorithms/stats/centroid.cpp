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

#include "centroid.h"

using namespace essentia;
using namespace standard;

const char* Centroid::name = "Centroid";
const char* Centroid::category = "Statistics";
const char* Centroid::description = DOC("This algorithm computes the centroid of an array. The centroid is normalized to a specified range. This algorithm can be used to compute spectral centroid or temporal centroid.\n"
"\n"
"The spectral centroid is a measure that indicates where the \"center of mass\" of the spectrum is. Perceptually, it has a robust connection with the impression of \"brightness\" of a sound, and therefore is used to characterise musical timbre. It is calculated as the weighted mean of the frequencies present in the signal, with their magnitudes as the weights.\n"
"\n"
"The temporal centroid is the point in time in a signal that is a temporal balancing point of the sound event energy. It can be computed from the envelope of the signal across audio samples [3] (see Envelope algorithm) or over the RMS level of signal across frames [4] (see RMS algorithm).\n"
"\n"
"Note:\n"
"- For a spectral centroid [hz], frequency range should be equal to samplerate/2\n"
"- For a temporal envelope centroid [s], range should be equal to (audio_size_in_samples-1) / samplerate\n"
"- Exceptions are thrown when input array contains less than 2 elements.\n"
"\n"
"References:\n"
"  [1] Function Centroid -- from Wolfram MathWorld,\n"
"  http://mathworld.wolfram.com/FunctionCentroid.html\n"
"  [2] Spectral centroid - Wikipedia, the free encyclopedia,\n"
"  https://en.wikipedia.org/wiki/Spectral_centroid\n"
"  [3] G. Peeters, \"A large set of audio features for sound description\n"     
"  (similarity and classification) in the CUIDADO project,\" CUIDADO I.S.T.\n"  
"  Project Report, 2004.\n" 
"  [4] Klapuri, A., & Davy, M. (Eds.). (2007). Signal processing methods for\n"
"  music transcription. Springer Science & Business Media.");


void Centroid::configure() {
  // get the range parameter as a Real (its native type) in the configure()
  // method instead of the compute() one, so we just need to do this once when
  // the object is configured, and not each time we call the compute() method.
  _range = parameter("range").toReal();
}

void Centroid::compute() {

  const std::vector<Real>& array = _array.get();
  Real& centroid = _centroid.get();

  if (array.empty()) {
    throw EssentiaException("Centroid: cannot compute the centroid of an empty array");
  }

  if (array.size() == 1) {
    throw EssentiaException("Centroid: cannot compute the centroid of an array of size 1");
  }

  centroid = 0.0;
  Real weights = 0.0;

  for (int i=0; i<int(array.size()); ++i) {
    centroid += i * array[i];
    weights += array[i];
  }

  if (weights != 0.0) {
    centroid /= weights;
  }
  else {
    centroid = 0.0;
  }

  centroid *= _range / (array.size() - 1);
}
