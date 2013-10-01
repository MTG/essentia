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

#include "rms.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* RMS::name = "RMS";
const char* RMS::description = DOC("This algorithm computes the Root Mean Square (quadratic mean) of an array of Reals.\n"
"RMS is not defined for empty arrays. In such case, an exception will be thrown\n."
"\n"
"References:\n"
"  [1] Root mean square - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Root_mean_square");

void RMS::compute() {

  const std::vector<Real>& array = _array.get();
  Real& rms = _rms.get();

  if (array.empty()) {
    throw EssentiaException("RMS: input array is empty");
  }

  rms = 0.0;

  for (int i=0; i<int(array.size()); ++i) {
    rms += array[i]*array[i];
  }

  rms /= array.size();
  rms = sqrt(rms);
}
