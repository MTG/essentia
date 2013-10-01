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

#include "geometricmean.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* GeometricMean::name = "GeometricMean";
const char* GeometricMean::description = DOC("This algorithm computes the geometric mean of an array of positive Reals.\n"
"\n"
"An exception is thrown if the input array does not contain strict positive numbers or the array is empty.\n"
"\n"
"References:\n"
"  [1] Energy (signal processing) - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Energy_(signal_processing)\n\n"
"  [2] Geometric Mean -- from Wolfram MathWorld,\n"
"  http://mathworld.wolfram.com/GeometricMean.html");

void GeometricMean::compute() {

  const std::vector<Real>& array = _array.get();
  Real& geometricMean = _geometricMean.get();

  if (array.empty()) {
    throw EssentiaException("GeometricMean: input array empty, cannot compute the geometric mean of an empty array");
  }

  for (std::vector<Real>::size_type i=0; i<array.size(); i++) {
    if (array[i] < 0) {
      throw EssentiaException("GeometricMean: input array contains negative numbers");
    }
  }


  geometricMean = 0.0;

  for (std::vector<Real>::size_type i=0; i<array.size(); i++) {
    if (array[i] == 0.0) {
      geometricMean = 0.0;
      return;
    }
    else {
      geometricMean += log(array[i]);
    }
  }

  geometricMean /= (Real)array.size();

  geometricMean = exp(geometricMean);
}
