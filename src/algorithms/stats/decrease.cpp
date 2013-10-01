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

#include "decrease.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* Decrease::name = "Decrease";
const char* Decrease::description = DOC("This algorithm extracts the decrease of an array of Reals (which is defined as the linear regression coefficient). The range parameter is used to normalize the result. For a spectral centroid, the range should be equal to Nyquist and for an audio centroid the range should be equal to (audiosize - 1) / samplerate.\n"
"The size of the input array must be at least two elements for \"decrease\" to be computed, otherwise an exception is thrown.\n"
"References:\n"
"  [1] Least Squares Fitting -- from Wolfram MathWorld,\n"
"  http://mathworld.wolfram.com/LeastSquaresFitting.html"
);

void Decrease::configure() {
  _range = parameter("range").toReal();

  if (_range == 0) throw EssentiaException("Decrease: range parameter cannot be zero");
}

void Decrease::compute() {
  const std::vector<Real>& array = _array.get();
  Real& decrease = _decrease.get();

  if (array.size() <= 1) {
    throw EssentiaException("Decrease: array size must be greater than 1");
  }

  Real scaler = _range / (array.size() - 1.0);

  Real mean_x = _range / 2.0;
  Real mean_y = mean(array);

  Real ss_xx = 0.0;
  Real ss_xy = 0.0;
  for (int i=0; i<int(array.size()); ++i) {
    Real tmp = Real(i) * scaler - mean_x;
    ss_xx += tmp * tmp;
    ss_xy += tmp * (array[i] - mean_y);
  }

  decrease = ss_xy / ss_xx;
}
