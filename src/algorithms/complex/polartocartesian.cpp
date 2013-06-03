/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
"      http://en.wikipedia.org/wiki/Polar_coordinates");

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
