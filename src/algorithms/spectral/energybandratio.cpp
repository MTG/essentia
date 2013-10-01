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
"  http://en.wikipedia.org/wiki/Energy_(signal_processing)");


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
