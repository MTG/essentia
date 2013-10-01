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

#include "instantpower.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* InstantPower::name = "InstantPower";
const char* InstantPower::description = DOC("This algorithm computes the instant power of an array. That is, the energy of the array over its size.\n"
"\n"
"An exception is thrown when input array is empty.\n"
"\n"
"References:\n"
"  [1] Energy (signal processing) - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Energy_(signal_processing)");

void InstantPower::compute() {
  const std::vector<Real>& array = _array.get();
  if (array.empty()) {
    throw EssentiaException("InstantPower: cannot compute the instant power of an empty array");
  }

  _power.get() = instantPower(array);
}
