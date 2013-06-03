/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "variance.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* Variance::name = "Variance";
const char* Variance::description = DOC(
  "This algorithm calculates the variance of an array of Reals.");

void Variance::compute() {
  _variance.get() = variance(_array.get(),mean(_array.get()));
}
