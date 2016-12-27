/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#include "loudness.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* Loudness::name = "Loudness";
const char* Loudness::category = "Loudness/dynamics";
const char* Loudness::description = DOC("This algorithm computes the loudness of an audio signal defined by Steven's power law. It computes loudness as the energy of the signal raised to the power of 0.67.\n"
"\n"
"References:\n"
"  [1] Energy (signal processing) - Wikipedia, the free encyclopedia\n"
"  http://en.wikipedia.org/wiki/Energy_(signal_processing)\n\n"
"  [2] Stevens' power law - Wikipedia, the free encyclopedia\n"
"  http://en.wikipedia.org/wiki/Stevens%27_power_law\n\n"
"  [3] S. S. Stevens, Psychophysics. Transaction Publishers, 1975.");


void Loudness::compute() {

  const std::vector<Real>& signal = _signal.get();
  Real& loudness = _loudness.get();

  Real signalEnergy = energy(signal);

  loudness = powf(signalEnergy, 0.67);
}
