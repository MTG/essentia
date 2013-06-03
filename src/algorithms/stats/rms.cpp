/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "rms.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* RMS::name = "RMS";
const char* RMS::description = DOC("This algorithm computes the Root Mean Square (quadratic mean) of an array of Reals.\n"
"RMS is not defined for empty arrays. In such case, an exception will be thrown\n."
"\n"
"References:\n"
"  [1] Root mean square - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Root_mean_square");

void RMS::compute() {

  const std::vector<Real>& array = _array.get();
  Real& rms = _rms.get();

  if (array.empty()) {
    throw EssentiaException("RMS: input array is empty");
  }

  rms = 0.0;

  for (int i=0; i<int(array.size()); ++i) {
    rms += array[i]*array[i];
  }

  rms /= array.size();
  rms = sqrt(rms);
}
