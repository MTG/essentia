/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
"  [1] Moving Average Filters,\n"
"      http://www.dspguide.com/ch15.htm");


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
