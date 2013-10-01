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
"  John Wiley & Sons, 2002");


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
