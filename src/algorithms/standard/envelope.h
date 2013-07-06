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

#ifndef ESSENTIA_ENVELOPE_H
#define ESSENTIA_ENVELOPE_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Envelope : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _envelope;

 public:
  Envelope() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_envelope, "signal", "the resulting envelope of the signal");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the audio sampling rate [Hz]", "(0,inf)", 44100.);
    declareParameter("attackTime", "the attack time of the first order lowpass in the attack phase [ms]", "[0,inf)", 10.0);
    declareParameter("releaseTime", "the release time of the first order lowpass in the release phase [ms]", "[0,inf)", 1500.0);
    declareParameter("applyRectification", "whether to apply rectification (envelope based on the absolute value of signal)", "{true,false}", true);
  }

  void configure();
  void reset();
  void compute();

  static const char* name;
  static const char* description;

 protected:
  // output of the filter
  Real _tmp;

  // attack and release coefficient for the filter
  Real _ga;
  Real _gr;

  bool _applyRectification;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Envelope : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _signal;
  Source<Real> _envelope;

 public:
  Envelope() {
    declareAlgorithm("Envelope");
    declareInput(_signal, STREAM, 4096, "signal");
    declareOutput(_envelope, STREAM, 4096, "signal");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_ENVELOPE_H
