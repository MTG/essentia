/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "zerocrossingrate.h"
#include <cmath>

using namespace std;
using namespace essentia;
using namespace standard;


const char* ZeroCrossingRate::name = "ZeroCrossingRate";
const char* ZeroCrossingRate::description = DOC(
"This algorithm returns the zero-crossing rate of an audio signal. It is the number of sign changes between consecutive signal values divided by the total number of values. Noisy signals tend to have higher zero-crossing rate.\n"
"In order to avoid small variations around zero caused by noise, a threshold around zero is given to consider a valid zerocrosing whenever the boundary is crossed.\n"
"Empty input signals will raise an exception.\n"
"References:\n"
"  [1] Zero Crossing - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Zero-crossing_rate\n"
"  [2] G. Peeters, A large set of audio features for sound description (similarity and classification) in the CUIDADO project,"
"      CUIDADO I.S.T. Project Report, 2004");


void ZeroCrossingRate::configure() {
  _threshold = fabs(parameter("threshold").toReal());
}

void ZeroCrossingRate::compute() {

  const std::vector<Real>& signal = _signal.get();
  Real& zeroCrossingRate = _zeroCrossingRate.get();

  if (signal.empty()) throw EssentiaException("ZeroCrossingRate: the input signal is empty");

  zeroCrossingRate = 0.0;
  Real val = signal[0];
  if (std::fabs(val) < _threshold) val = 0;
  bool was_positive = (val > 0.0 );
  bool is_positive;

  for (int i=1; i<int(signal.size()); i++) {
    val = signal[i];
    if (std::fabs(val) <= _threshold) val = 0;
    is_positive = val > 0.0;
    if (was_positive != is_positive) {
      zeroCrossingRate++;
      was_positive = is_positive;
    }
  }

  zeroCrossingRate /= signal.size();
}

