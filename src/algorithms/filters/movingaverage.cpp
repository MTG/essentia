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

#include "movingaverage.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;
using namespace std;

const char* MovingAverage::name = "MovingAverage";
const char* MovingAverage::description = DOC("This algorithm implements an FIR Moving Average filter. Because of its dependece on IIR, IIR's requirements are inherited.\n"
"\n"
"References:\n"
"  [1] Moving Average Filters, http://www.dspguide.com/ch15.htm");


void MovingAverage::configure() {
  int delay = parameter("size").toInt();

  vector<Real> b(delay, 1.0/delay);
  vector<Real> a(1, 1.0);

  _filter->configure("numerator", b, "denominator", a);
}

void MovingAverage::compute() {
  _filter->input("signal").set(_x.get());
  _filter->output("signal").set(_y.get());
  _filter->compute();
}
