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

#include "maxmagfreq.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* MaxMagFreq::name = "MaxMagFreq";
const char* MaxMagFreq::category = "Spectral";
const char* MaxMagFreq::description = DOC("This algorithm computes the frequency with the largest magnitude in a spectrum.\n"
"Note that a spectrum must contain at least two elements otherwise an exception is thrown");

void MaxMagFreq::compute() {

  const std::vector<Real>& spectrum = _spectrum.get();
  Real& maxMagFreq = _maxMagFreq.get();

  if (spectrum.size() < 2) {
    throw EssentiaException("MaxMagFreq: input audio spectrum must be larger than 1 element");
  }

  int index = std::max_element(spectrum.begin(), spectrum.end()) - spectrum.begin();

  // normalize the maximum to the desired frequency range
  // (be careful not to confuse with the sampling rate which is the double)
  maxMagFreq = index * (_sampleRate/2.0) / (spectrum.size()-1);
}
