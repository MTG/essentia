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

#include "magnitude.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* Magnitude::name = "Magnitude";
const char* Magnitude::description = DOC("This algorithm computes the absolute value of each element in a vector of complex numbers.\n"
"\n"
"References:\n"
"  [1] Complex Modulus -- from Wolfram MathWorld,\n"
"  http://mathworld.wolfram.com/ComplexModulus.html\n\n"
"  [2] Complex number - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Complex_numbers#Absolute_value.2C_conjugation_and_distance.");

void Magnitude::compute() {

  const std::vector<std::complex<Real> >& cmplex = _complex.get();
  std::vector<Real>& magnitude = _magnitude.get();

  magnitude.resize(cmplex.size());

  for (std::vector<Real>::size_type i=0; i<magnitude.size(); i++) {
    magnitude[i] = sqrt(cmplex[i].real()*cmplex[i].real() + cmplex[i].imag()*cmplex[i].imag());
  }
}
