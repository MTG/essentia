/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#include "bandreject.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;
using namespace std;

const char* BandReject::name = "BandReject";
const char* BandReject::category = "Filters";
const char* BandReject::description = DOC("This algorithm implements a 2nd order IIR band-reject filter. Because of its dependence on IIR, IIR's requirements are inherited.\n"
"\n"
"References:\n"
"  [1] U. Zölzer, DAFX - Digital Audio Effects, 2nd edition, p. 55,\n"
"  John Wiley & Sons, 2011");


void BandReject::configure() {
  Real fs = parameter("sampleRate").toReal();
  Real fc = parameter("cutoffFrequency").toReal();
  Real fb = parameter("bandwidth").toReal();

  Real c = (tan(M_PI*fb/fs) - 1) / (tan(M_PI*fb/fs) + 1);
  Real d = -cos(2*M_PI*fc/fs);

  vector<Real> b(3, 0.0);
  b[0] = (1.0-c)/2.0;
  b[1] = d*(1.0-c);
  b[2] = (1.0-c)/2.0;

  vector<Real> a(3, 0.0);
  a[0] = 1.0;
  a[1] = d*(1.0-c);
  a[2] = -c;

  _filter->configure( "numerator", b, "denominator", a);
}

void BandReject::compute() {
  _filter->input("signal").set(_x.get());
  _filter->output("signal").set(_y.get());
  _filter->compute();
}
