/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "loudness.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* Loudness::name = "Loudness";
const char* Loudness::description = DOC("This algorithm extracts the loudness of a signal, which is defined by Steven's power law as its energy raised to the power of 0.67.\n"
"\n"
"References:\n"
"  [1] Energy (signal processing) - Wikipedia, the free encyclopedia\n"
"      http://en.wikipedia.org/wiki/Energy_(signal_processing)"
"  [2] Stevens' power law - Wikipedia, the free encyclopedia\n" 
"      http://en.wikipedia.org/wiki/Stevens%27_power_law\n"
"  [3] S. S. Stevens, Psychophysics. Transaction Publishers, 1975.");


void Loudness::compute() {

  const std::vector<Real>& signal = _signal.get();
  Real& loudness = _loudness.get();

  Real signalEnergy = energy(signal);

  loudness = powf(signalEnergy, 0.67);
}
