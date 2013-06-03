/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "powermean.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* PowerMean::name = "PowerMean";
const char* PowerMean::description = DOC("This algorithm extracts the Power Mean of an array of Reals. It accepts one parameter, p, which is the power (or order or degree) of the Power Mean. Note that if p=-1, the Power Mean is equal to the Harmonic Mean, if p=0, the Power Mean is equal to the Geometric Mean, if p=1, the Power Mean is equal to the Arithmetic Mean, if p=2, the Power Mean is equal to the Root Mean Square.\n"
"\n"
"Exceptions are thrown if input array either is empty or it contains non positive numbers.\n"
"\n"
"References:\n"
"  [1] Power Mean -- from Wolfram MathWorld,\n"
"      http://mathworld.wolfram.com/PowerMean.html");

void PowerMean::compute() {

  const std::vector<Real>& array = _array.get();
  Real& powerMean = _powerMean.get();

  if (array.empty()) throw EssentiaException("PowerMean: input array is empty");

  powerMean = 0.0;

  Real p = parameter("power").toReal();

  if (p == 0.0) {

    _geometricMean->input("array").set(array);
    _geometricMean->output("geometricMean").set(powerMean);
    _geometricMean->compute();
  }
  else {
    for (int i = 0; i < int(array.size()); ++i) {
      if (array[i] < 0.0) {
        throw EssentiaException("PowerMean: input array contains non-positive real numbers (e.g. ", array[i], ")");
      }
      powerMean += powf(array[i], p);
    }

    powerMean /= array.size();

    powerMean = powf(powerMean, 1.0/p);
  }
}
