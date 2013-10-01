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

#include "flatnessdb.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* FlatnessDB::name = "FlatnessDB";
const char* FlatnessDB::description = DOC("This algorithm computes the flatness of an array, which is defined as the ratio between the geometric mean and the arithmetic mean and it converts it to dB scale."
"\n"
"The size of the input array must be greater than 0. If the input array is empty an exception will be thrown. This algorithm uses the Flatness algorithm and thus inherits its input requirements and exceptions.\n"
"\n"
"References:\n"
"  [1] G. Peeters, \"A large set of audio features for sound description\n"
"  (similarity and classification) in the CUIDADO project,\" CUIDADO I.S.T.\n"
"  Project Report, 2004");

void FlatnessDB::compute() {

  const std::vector<Real>& array = _array.get();

  if (array.empty()) {
    throw EssentiaException("FlatnessDB: size of input array is zero");
  }

  Real& flatnessDB = _flatnessDB.get();

  Real flatness;

  _flatness->input("array").set(array);
  _flatness->output("flatness").set(flatness);
  _flatness->compute();

  if (flatness <= 0.0) {
    flatnessDB = 1.0; // default value chosen for silent signals
  }
  else {
    flatnessDB = std::min( Real(lin2db(flatness)/-60.0), Real(1.0));
  }
}
