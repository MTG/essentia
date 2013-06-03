/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "maxmagfreq.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* MaxMagFreq::name = "MaxMagFreq";
const char* MaxMagFreq::description = DOC("This algorithm computes the frequency with the largest magnitude.\n"
"Note that a spectrum must contain at least two elements otherwise an exception is thrown");

void MaxMagFreq::compute() {

  const std::vector<Real>& spectrum = _spectrum.get();
  Real& maxMagFreq = _maxMagFreq.get();

  if (spectrum.size() < 2) {
    throw EssentiaException("MaxMagFreq: input audio spectrum must be larger than 1 element");
  }

  int index = std::max_element(spectrum.begin(), spectrum.end()) - spectrum.begin();

  // normalize the maximum to the desired frequency range
  // (be careful not to confuse with the sampling rate which is the double)
  maxMagFreq = index * (_sampleRate/2.0) / (spectrum.size()-1);
}
