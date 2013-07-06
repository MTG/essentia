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

#ifndef ESSENTIA_LARM_H
#define ESSENTIA_LARM_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class Larm : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<Real> _larm;
  Algorithm* _envelope;
  Algorithm* _powerMean;

 public:
  Larm() {
    declareInput(_signal, "signal", "the audio input signal");
    declareOutput(_larm, "larm", "the LARM loudness estimate [dB]");

    _envelope = AlgorithmFactory::create("Envelope");
    _powerMean = AlgorithmFactory::create("PowerMean");
  }

  ~Larm() {
    delete _envelope;
    delete _powerMean;
  }

  void reset() {
    _envelope->reset();
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("attackTime", "the attack time of the first order lowpass in the attack phase [ms]", "[0,inf)", 10.0);
    declareParameter("releaseTime", "the release time of the first order lowpass in the release phase [ms]", "[0,inf)", 1500.0);
    declareParameter("power", "the power used for averaging", "(-inf,inf)", 1.5); // 1.5 is an empirical value, see the paper...
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

class Larm : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<Real> _larm;

 public:
  Larm() {
    declareAlgorithm("Larm");
    declareInput(_signal, TOKEN, "signal");
    declareOutput(_larm, TOKEN, "larm");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_LARM_H
