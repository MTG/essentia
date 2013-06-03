/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "energyband.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* EnergyBand::name = "EnergyBand";
const char* EnergyBand::description = DOC("This algorithm computes the spectral energy of the given frequency band, including both start and stop cutoff frequencies.\n"
"Note that exceptions will be thrown when input spectrum is empty and if startCutoffFrequency is greater than startCutoffFrequency.\n"
"References:\n"
"  [1] Energy (signal processing) - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Energy_(signal_processing)");

void EnergyBand::configure() {
  Real startFreq  = parameter("startCutoffFrequency").toReal();
  Real stopFreq   = parameter("stopCutoffFrequency").toReal();
  Real sampleRate = parameter("sampleRate").toReal();

  if (startFreq >= stopFreq) {
    throw EssentiaException("EnergyBand: stopCutoffFrequency must be larger than startCutoffFrequency");
  }

  Real nyquist=sampleRate/2.0;

  if (startFreq >= nyquist) {
    throw EssentiaException("EnergyBand: start frequency must be below the Nyquist frequency", nyquist);
  }
  if (stopFreq > nyquist) {
    throw EssentiaException("EnergyBand: stop frequency must be below or equal to the Nyquist frequency", nyquist);
  }

  _normStartIdx = startFreq/nyquist;
  _normStopIdx  = stopFreq /nyquist;
}

void EnergyBand::compute() {
  const std::vector<Real>& spectrum = _spectrum.get();
  Real& energyBand = _energyBand.get();
  if (spectrum.empty()) {
    throw EssentiaException("EnergyBand: spectrum is empty");
  }

  // start/stop is the index corresponding to the start/stop cut-off frequency
  int start = int(round(_normStartIdx * (spectrum.size() - 1)));
  int stop  = int(round(_normStopIdx  * (spectrum.size() - 1)));

  energyBand = 0.0;

  for (int i=start; i<=stop; ++i) {
    energyBand += spectrum[i]*spectrum[i];
  }
}
