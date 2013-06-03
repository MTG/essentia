/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "noiseadder.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* NoiseAdder::name = "NoiseAdder";
const char* NoiseAdder::description = DOC("This algorithm adds some amount of noise to an input signal, and returns the resulting output signal. The average energy of the noise in dB is defined by the level parameter, and is generated using the Mersenne Twister random number generator.\n"
"\n"
"References:\n"
"  [1] Mersenne Twister: A random number generator (since 1997/10),\n"
"      http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html\n"
"  [2] Mersenne twister - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Mersenne_twister");


// The implementation uses the MersenneTwister.h file downloaded from:
// http://www-personal.engin.umich.edu/~wagnerr/MersenneTwister.html

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
    noise[i] = signal[i] + _level * ((Real)_mtrand()*2.0 - 1.0);
  }

}
