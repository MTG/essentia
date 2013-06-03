/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "lowpass.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;
using namespace std;

const char* LowPass::name = "LowPass";
const char* LowPass::description = DOC("This algorithm implements a 1st order IIR low-pass filter. Because of its dependence on IIR, IIR's requirements are inherited.\n"
"References:\n"
"  [1] U. Zölzer, DAFX - Digital Audio Effects, p. 40,\n"
"      John Wiley & Sons, 2002");


void LowPass::configure() {

  Real fs = parameter("sampleRate").toReal();
  Real fc = parameter("cutoffFrequency").toReal();

  Real c = (tan(M_PI*fc/fs) - 1) /
           (tan(M_PI*fc/fs) + 1);

  vector<Real> b(2, 0.0);
  b[0] = (1.0+c)/2.0;
  b[1] = (1.0+c)/2.0;

  vector<Real> a(2, 0.0);
  a[0] = 1.0;
  a[1] = c;

  _filter->configure("numerator", b, "denominator", a);
}

void LowPass::compute() {
  _filter->input("signal").set(_x.get());
  _filter->output("signal").set(_y.get());
  _filter->compute();
}
