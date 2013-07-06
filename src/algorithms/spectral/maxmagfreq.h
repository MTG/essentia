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

#ifndef ESSENTIA_MAXMAGFREQ_H
#define ESSENTIA_MAXMAGFREQ_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class MaxMagFreq : public Algorithm {

 private:
  Input<std::vector<Real> > _spectrum;
  Output<Real> _maxMagFreq;
  Real _sampleRate;

 public:
  MaxMagFreq() {
    declareInput(_spectrum, "spectrum", "the input spectrum (must have more than 1 element)");
    declareOutput(_maxMagFreq, "maxMagFreq", "the frequency with the largest magnitude [Hz]");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the audio sampling rate [Hz]", "(0,inf)", 44100.);
  }

  void configure() {
    _sampleRate = parameter("sampleRate").toReal();
  }

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class MaxMagFreq : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _maxMagFreq;

 public:
  MaxMagFreq() {
    declareAlgorithm("MaxMagFreq");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_maxMagFreq, TOKEN, "maxMagFreq");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_MAXMAGFREQ_H
