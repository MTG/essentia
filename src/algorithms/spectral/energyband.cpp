/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of Essentia
 *
 * Essentia is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 */

#include "energyband.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* EnergyBand::name = "EnergyBand";
const char* EnergyBand::description = DOC("This algorithm computes the spectral energy of the given frequency band, including both start and stop cutoff frequencies.\n"
"Note that exceptions will be thrown when input spectrum is empty and if startCutoffFrequency is greater than startCutoffFrequency.\n"
"\n"
"References:\n"
"  [1] Energy (signal processing) - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Energy_(signal_processing)");

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
