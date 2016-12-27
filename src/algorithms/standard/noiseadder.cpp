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

#include "noiseadder.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* NoiseAdder::name = "NoiseAdder";
const char* NoiseAdder::category = "Standard";
const char* NoiseAdder::description = DOC("This algorithm adds noise to an input signal. The average energy of the noise in dB is defined by the level parameter, and is generated using the Mersenne Twister random number generator.\n"
"\n"
"References:\n"
"  [1] Mersenne Twister: A random number generator (since 1997/10),\n"
"  http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html\n\n"
"  [2] Mersenne twister - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Mersenne_twister");


void NoiseAdder::configure() {
  _level = db2pow(parameter("level").toReal());
  if (parameter("fixSeed").toBool()) {
    unsigned long seed = 0;
    _mtrand.seed(seed);
  }

}

void NoiseAdder::compute() {
  const std::vector<Real>& signal = _signal.get();
  std::vector<Real>& noise = _noise.get();

  std::vector<Real>::size_type size = signal.size();
  noise.resize(size);
  for (std::vector<Real>::size_type i=0; i<size; i++) {
#ifdef CPP_11
    noise[i] = signal[i] + _level * (Real(_mtrand())/std::mt19937::max()*2.0f - 1.0f);
#else
    noise[i] = signal[i] + _level * (Real(_mtrand())*2.0f - 1.0f);
#endif
  }

}
