/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "energybandratio.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* EnergyBandRatio::name = "EnergyBandRatio";
const char* EnergyBandRatio::description = DOC("This algorithm computes the ratio of the spectral energy in the range [startFrequency, stopFrequency] over the total energy.\n"
"\n"
"An exception is thrown when startFrequency is larger than stopFrequency\n"
"or the input spectrum is empty.\n"
"\n"
"References:\n"
"  [1] Energy (signal processing) - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Energy_(signal_processing)");


void EnergyBandRatio::configure() {
  Real freqRange = parameter("sampleRate").toReal() / 2.0;
  Real startFreq = parameter("startFrequency").toReal();
  Real stopFreq = parameter("stopFrequency").toReal();

  if (stopFreq < startFreq) {
    throw EssentiaException("EnergyBandRatio: stopFrequency is less than startFrequency");
  }

  _startFreqNormalized = startFreq / freqRange;
  _stopFreqNormalized = stopFreq / freqRange;
}


void EnergyBandRatio::compute() {

  const vector<Real>& spectrum = _spectrum.get();

  if (spectrum.empty()) {
    throw EssentiaException("EnergyBandRatio: input audio spectrum empty");
  }

  Real& energyBandRatio = _energyBandRatio.get();

  Real totalEnergy = energy(spectrum);

  if (totalEnergy <= 1e-10) {
    energyBandRatio = 0.0;
    return;
  }

  int start = int(_startFreqNormalized * (spectrum.size()-1) + 0.5);
  int stop = int(_stopFreqNormalized * (spectrum.size()-1) + 0.5) + 1;
  // +1 because we then loop with i<stopFreq instead of i<=stopFreq, so that if
  // start and stop are both > range, we get 0 energy (instead of energy of nyquist freq)

  if (start < 0) start = 0;
  if (stop > int(spectrum.size())) stop = spectrum.size();

  Real energy = 0.0;

  for (int i = start; i < stop; i++) {
    energy += spectrum[i]*spectrum[i];
  }

  energyBandRatio = energy / totalEnergy;
}
