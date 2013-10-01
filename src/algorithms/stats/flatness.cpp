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

#include "flatness.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* Flatness::name = "Flatness";
const char* Flatness::description = DOC("This algorithm computes the flatness of an array, which is defined as the ratio between the geometric mean and the arithmetic mean.\n"
"\n"
"Flatness is undefined for empty input and negative values, therefore an exception is thrown in any both cases.\n"
"\n"
"References:\n"
"  [1] G. Peeters, \"A large set of audio features for sound description\n"
"  (similarity and classification) in the CUIDADO project,\" CUIDADO I.S.T.\n"
"  Project Report, 2004");

void Flatness::compute() {

  const std::vector<Real>& array = _array.get();

  if (array.size() == 0) {
    throw EssentiaException("Flatness: the input array has size zero");
  }

  for (std::vector<Real>::size_type i=0; i<array.size(); i++) {
    if (array[i] < 0) {
      throw EssentiaException("Flatness: the input array has negative values");
    }
  }

  Real& flatness = _flatness.get();

  Real geometricMean;

  _geometricMean->input("array").set(array);
  _geometricMean->output("geometricMean").set(geometricMean);
  _geometricMean->compute();

  if (geometricMean == 0.0) {
    flatness = 0.0;
  }
  else {
    Real arithmeticMean = mean(array);
    // this division never fails, because as the geometric mean is > 0, it means
    // that all values in the array are > 0, hence it is impossible for the
    // arithmetic mean to be 0
    flatness = geometricMean / arithmeticMean;
  }
}
