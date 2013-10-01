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

#include "polartocartesian.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* PolarToCartesian::name = "PolarToCartesian";
const char* PolarToCartesian::description = DOC("This algorithm converts an array of complex numbers from its polar form to its cartesian form through the Euler formula:\n"
"  z = x + i*y = |z|(cos(α) + i sin(α))\n"
"    where x = Real part, y = Imaginary part,\n"
"    and |z| = modulus = magnitude, α = phase\n"
"\n"
"An exception is thrown if the size of the magnitude vector does not match the size of the phase vector.\n"
"\n"
"References:\n"
"  [1] Polar coordinate system - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Polar_coordinates");

void PolarToCartesian::compute() {

  const vector<Real>& magnitude = _magnitude.get();
  const vector<Real>& phase = _phase.get();
  vector<complex<Real> >& complexVec = _complex.get();

  if (magnitude.size() != phase.size()) {
    ostringstream msg;
    msg << "PolarToCartesian: Could not merge magnitude array (size " << magnitude.size()
        << ") with phase array (size " << phase.size() << ") because of their different sizes";
    throw EssentiaException(msg);
  }

  complexVec.resize(magnitude.size());

  for (int i=0; i<int(magnitude.size()); ++i) {
    complexVec[i] = complex<Real>(magnitude[i] * cos(phase[i]),
                                  magnitude[i] * sin(phase[i]));
  }
}
