/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "geometricmean.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* GeometricMean::name = "GeometricMean";
const char* GeometricMean::description = DOC("This algorithm computes the geometric mean of an array of positive Reals.\n"
"\n"
"An exception is thrown if the input array does not contain strict positive numbers or the array is empty.\n"
"\n"
"References:\n"
"  [1] Energy (signal processing) - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Energy_(signal_processing)\n"
"  [2] Geometric Mean -- from Wolfram MathWorld,\n"
"      http://mathworld.wolfram.com/GeometricMean.html");

void GeometricMean::compute() {

  const std::vector<Real>& array = _array.get();
  Real& geometricMean = _geometricMean.get();

  if (array.empty()) {
    throw EssentiaException("GeometricMean: input array empty, cannot compute the geometric mean of an empty array");
  }

  for (std::vector<Real>::size_type i=0; i<array.size(); i++) {
    if (array[i] < 0) {
      throw EssentiaException("GeometricMean: input array contains negative numbers");
    }
  }


  geometricMean = 0.0;

  for (std::vector<Real>::size_type i=0; i<array.size(); i++) {
    if (array[i] == 0.0) {
      geometricMean = 0.0;
      return;
    }
    else {
      geometricMean += log(array[i]);
    }
  }

  geometricMean /= (Real)array.size();

  geometricMean = exp(geometricMean);
}
