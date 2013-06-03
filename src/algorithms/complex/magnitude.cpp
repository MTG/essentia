/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
"      http://mathworld.wolfram.com/ComplexModulus.html\n"
"  [2] Complex number - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Complex_numbers#Absolute_value.2C_conjugation_and_distance.");

void Magnitude::compute() {

  const std::vector<std::complex<Real> >& cmplex = _complex.get();
  std::vector<Real>& magnitude = _magnitude.get();

  magnitude.resize(cmplex.size());

  for (std::vector<Real>::size_type i=0; i<magnitude.size(); i++) {
    magnitude[i] = sqrt(cmplex[i].real()*cmplex[i].real() + cmplex[i].imag()*cmplex[i].imag());
  }
}
