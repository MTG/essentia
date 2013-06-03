/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "instantpower.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* InstantPower::name = "InstantPower";
const char* InstantPower::description = DOC("This algorithm computes the instant power of an array. That is, the energy of the array over its size.\n"
"\n"
"An exception is thrown when input array is empty.\n"
"\n"
"References:\n"
"  [1] Energy (signal processing) - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Energy_(signal_processing)");

void InstantPower::compute() {
  const std::vector<Real>& array = _array.get();
  if (array.empty()) {
    throw EssentiaException("InstantPower: cannot compute the instant power of an empty array");
  }

  _power.get() = instantPower(array);
}
