/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "mean.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* Mean::name = "Mean";
const char* Mean::description = DOC("This algorithm extracts the mean of an array of Reals.");

void Mean::compute() {
  _mean.get() = mean(_array.get());
}
