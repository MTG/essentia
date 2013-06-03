/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "bandpass.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;
using namespace std;

const char* BandPass::name = "BandPass";
const char* BandPass::description = DOC("This algorithm implements a 2nd order IIR band-pass filter. Because of its dependence on IIR, IIR's requirements are inherited.\n"
"\n"
"References:\n"
"  [1] U. Zölzer, DAFX - Digital Audio Effects, p. 43,\n"
"      John Wiley & Sons, 2002");


void BandPass::configure() {
  Real fs = parameter("sampleRate").toReal();
  Real fc = parameter("cutoffFrequency").toReal();
  Real fb = parameter("bandwidth").toReal();

  Real c = (tan(M_PI*fb/fs) - 1) / (tan(M_2PI*fb/fs) + 1);
  Real d = -cos(2*M_PI*fc/fs);

  vector<Real> b(3, 0.0);
  b[0] = (1.0+c)/2.0;
  b[1] = 0.0;
  b[2] = -(1.0+c)/2.0;

  vector<Real> a(3, 0.0);
  a[0] = 1.0;
  a[1] = d*(1.0-c);
  a[2] = -c;

  _filter->configure("numerator", b, "denominator", a);
}

void BandPass::compute() {
  _filter->input("signal").set(_x.get());
  _filter->output("signal").set(_y.get());
  _filter->compute();
}
