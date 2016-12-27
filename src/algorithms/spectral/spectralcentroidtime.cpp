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

#include "spectralcentroidtime.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* SpectralCentroidTime::name = "SpectralCentroidTime";
const char* SpectralCentroidTime::category = "Spectral";
const char* SpectralCentroidTime::description = DOC("This algorithm computes the spectral centroid of a signal in time domain. A first difference filter is applied to the input signal. Then the centroid is computed by dividing the norm of the resulting signal by the norm of the input signal. The centroid is given in hertz.\n"
 "References:\n"
 " [1] Udo Zölzer (2002). DAFX Digital Audio Effects pag.364-365\n");

void SpectralCentroidTime::configure() {
  _sampleRate = parameter("sampleRate").toReal();
}

void SpectralCentroidTime::compute() {

  const std::vector<Real>& signal = _signal.get();
  Real& centroid = _centroid.get();
  double a, b, aPowerSum = 0, bPowerSum = 0;

  if (signal.empty()) {
    throw EssentiaException("Centroid: cannot compute the centroid of an empty array");
  }

  if (signal.size() == 1) {
    throw EssentiaException("Centroid: cannot compute the centroid of an array of size 1");
  }

  for (int i=1; i<int(signal.size()); ++i) {
    a = signal[i];
    // first-difference filter
    b = signal[i] - signal[i-1];
    aPowerSum += a * a;
    bPowerSum += b * b;
  }
  // event from nan if input signal is empty
  if (bPowerSum == 0 || aPowerSum == 0) {
  centroid = 0;
  }
  else {
  centroid = (sqrt(bPowerSum)/sqrt(aPowerSum))*(_sampleRate/M_2PI);
  }

}