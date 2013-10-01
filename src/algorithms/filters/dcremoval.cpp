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

#include "dcremoval.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;
using namespace std;

const char* DCRemoval::name = "DCRemoval";
const char* DCRemoval::description = DOC("This algorithm removes the DC offset from a signal using a 1st order IIR highpass filter. Because of its dependence on IIR, IIR's requirements are inherited.\n"
"\n"
"References:\n"
"  [1] Smith, J.O.  Introduction to Digital Filters with Audio Applications,\n"
"  http://ccrma-www.stanford.edu/~jos/filters/DC_Blocker.html");


void DCRemoval::configure() {
  Real fs = parameter("sampleRate").toReal();
  Real fc = parameter("cutoffFrequency").toReal();
  // R is the feedback coefficient of the filter
  // we could not find a reference from where this formula came from, however
  // it fits very well to Julius O. Smith's observations of 0.995 for 44.1kHz
  // and scales with the sampling rate
  Real R = (Real)(1.0 - 2.0*M_PI*fc/fs);

  vector<Real> b(2, 0.0);
  b[0] = 1.0;
  b[1] = -1.0;

  vector<Real> a(2, 0.0);
  a[0] = 1.0;
  a[1] = -R;

  _filter->configure("numerator", b, "denominator", a);
}

void DCRemoval::compute() {
  _filter->input("signal").set(_signal.get());
  _filter->output("signal").set(_signalDC.get());
  _filter->compute();
}
