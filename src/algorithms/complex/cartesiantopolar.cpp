/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */


#include "cartesiantopolar.h"
#include "essentiamath.h"


using namespace essentia;
using namespace standard;


const char* CartesianToPolar::name = "CartesianToPolar";
const char* CartesianToPolar::description = DOC("This algorithm converts an array of complex numbers from its cartesian form to its polar form using the Euler formula:\n"
"  z = x + i*y = |z|(cos(α) + i sin(α))\n"
"    where x = Real part, y = Imaginary part,\n"
"    and |z| = modulus = magnitude, α = phase in (-pi,pi]\n"
"\n"
"It returns the magnitude and the phase as 2 separate vectors.\n"
"\n"
"References:\n"
"  [1] Polar Coordinates -- from Wolfram MathWorld,\n"
"      http://mathworld.wolfram.com/PolarCoordinates.html\n"
"  [2] Polar coordinate system - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Polar_coordinates");


void CartesianToPolar::compute() {

  const std::vector<std::complex<Real> >& c = _complex.get();
  std::vector<Real>& magnitude = _magnitude.get();
  std::vector<Real>& phase = _phase.get();

  magnitude.resize(c.size());
  phase.resize(c.size());

  for (std::vector<Real>::size_type i=0; i<magnitude.size(); i++) {
    magnitude[i] = sqrt(c[i].real()*c[i].real() + c[i].imag()*c[i].imag());
  }

  for (std::vector<Real>::size_type i=0; i<phase.size(); i++) {
    phase[i] = atan2(c[i].imag(), c[i].real());
  }
}
