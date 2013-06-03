/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "allpass.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;
using namespace std;

const char* AllPass::name = "AllPass";
const char* AllPass::description = DOC("This algorithm implements a IIR all-pass filter of order 1 or 2. Because of its dependence on IIR, IIR's requirements are inherited.\n"
"\n"
"References:\n"
"  [1] U. Zölzer, DAFX - Digital Audio Effects, p. 43,\n"
"      John Wiley & Sons, 2002");


void AllPass::configure() {
  Real fs = parameter("sampleRate").toReal();
  Real fc = parameter("cutoffFrequency").toReal();
  Real fb = parameter("bandwidth").toReal();
  int order = parameter("order").toInt();

  if (order == 1) {

    Real c = (tan(M_PI*fc/fs) - 1) / (tan(M_PI*fc/fs) + 1);

    vector<Real> b(2, 0.0);
    b[0] = c;
    b[1] = 1.0;

    vector<Real> a(2, 0.0);
    a[0] = 1.0;
    a[1] = c;

    _filter->configure( "numerator", b, "denominator", a);
  }
  else if (order == 2) {

    Real c = (tan(M_PI*fb/fs) - 1) / (tan(M_PI*fb/fs) + 1);
    Real d = -cos(2*M_PI*fc/fs);

    vector<Real> b(3, 0.0);
    b[0] = -c;
    b[1] = d*(1.0-c);
    b[2] = 1.0;

    vector<Real> a(3, 0.0);
    a[0] = 1.0;
    a[1] = d*(1.0-c);
    a[2] = -c;

    _filter->configure( "numerator", b, "denominator", a);
  }

}

void AllPass::compute() {
  _filter->input("signal").set(_x.get());
  _filter->output("signal").set(_y.get());
  _filter->compute();
}
