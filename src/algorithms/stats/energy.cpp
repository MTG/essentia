/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "energy.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* Energy::name = "Energy";
const char* Energy::description = DOC("This algorithm computes the energy of an array of Reals.\n"
"\n"
"The input array should not be empty or an exception will be thrown.\n"
"\n"
"References:\n"
"  [1] Energy (signal processing) - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Energy_(signal_processing)");

void Energy::compute() {
  const std::vector<Real>& array = _array.get();

  if (array.empty()) {
    throw EssentiaException("Energy: the input array size is zero");
  }

  _energy.get() = energy(array);
}
