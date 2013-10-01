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

#include "highpass.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;
using namespace std;

const char* HighPass::name = "HighPass";
const char* HighPass::description = DOC("This algorithm implements a 1st order IIR high-pass filter. Because of its dependence on IIR, IIR's requirements are inherited.\n"
"\n"
"References:\n"
"  [1] U. Zölzer, DAFX - Digital Audio Effects, p. 40,\n"
"  John Wiley & Sons, 2002");

void HighPass::configure() {

  Real fs = parameter("sampleRate").toReal();
  Real fc = parameter("cutoffFrequency").toReal();

  Real c = (tan(M_PI*fc/fs) - 1) / (tan(M_PI*fc/fs) + 1);

  vector<Real> b(2, 0.0);
  b[0] = (1.0-c)/2.0;
  b[1] = (c-1.0)/2.0;

  vector<Real> a(2, 0.0);
  a[0] = 1.0;
  a[1] = c;

  _filter->configure("numerator", b, "denominator", a);
}

void HighPass::compute() {
  _filter->input("signal").set(_x.get());
  _filter->output("signal").set(_y.get());
  _filter->compute();
}
