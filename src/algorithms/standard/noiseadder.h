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

#ifndef ESSENTIA_NOISEADDER_H
#define ESSENTIA_NOISEADDER_H

#include "algorithm.h"
#include "MersenneTwister.h"

namespace essentia {
namespace standard {

class NoiseAdder : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _noise;

  MTRand _mtrand;
  Real _level;

 public:
  NoiseAdder() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_noise, "signal", "the output signal with the added noise");
  }

  void declareParameters() {
    declareParameter("level", "power level of the noise generator [dB]", "(-inf,0]", -100);
    declareParameter("fixSeed", "if true, 0 is used as the seed for generating random values", "{true,false}", false);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class NoiseAdder : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _signal;
  Source<Real> _noise;

 public:
  NoiseAdder() {
    int preferredSize = 4096; // arbitrary
    declareAlgorithm("NoiseAdder");
    declareInput(_signal, STREAM, preferredSize, "signal");
    declareOutput(_noise, STREAM, preferredSize, "signal");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_NOISEADDER_H
