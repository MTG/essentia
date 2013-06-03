/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "centroid.h"

using namespace essentia;
using namespace standard;

const char* Centroid::name = "Centroid";
const char* Centroid::description = DOC("This algorithm extracts the centroid (first order central moment), normalized to a specified range, of the input array.\n"
"Note:\n"
" - For a spectral centroid [hz], frequency range should be equal to samplerate/2\n"
" - For an audio centroid [s], frequency range should be equal to (audio_size-1) / samplerate\n"
"Exceptions are thrown when input array contains less than 2 elements.\n"
"\n"
"References:\n"
"  [1] Function Centroid -- from Wolfram MathWorld,\n"
"      http://mathworld.wolfram.com/FunctionCentroid.html");


void Centroid::configure() {
  // get the range parameter as a Real (its native type) in the configure()
  // method instead of the compute() one, so we just need to do this once when
  // the object is configured, and not each time we call the compute() method.
  _range = parameter("range").toReal();
}

void Centroid::compute() {

  const std::vector<Real>& array = _array.get();
  Real& centroid = _centroid.get();

  if (array.empty()) {
    throw EssentiaException("Centroid: cannot compute the centroid of an empty array");
  }

  if (array.size() == 1) {
    throw EssentiaException("Centroid: cannot compute the centroid of an array of size 1");
  }

  centroid = 0.0;
  Real weights = 0.0;

  for (int i=0; i<int(array.size()); ++i) {
    centroid += i * array[i];
    weights += array[i];
  }

  if (weights != 0.0) {
    centroid /= weights;
  }
  else {
    centroid = 0.0;
  }

  centroid *= _range / (array.size() - 1);
}
