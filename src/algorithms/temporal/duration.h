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

#ifndef ESSENTIA_DURATION_H
#define ESSENTIA_DURATION_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Duration : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<Real> _duration;

 public:
  Duration() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_duration, "duration", "the duration of the signal [s]");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
  }

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia


#include "accumulatoralgorithm.h"

namespace essentia {
namespace streaming {

class Duration : public AccumulatorAlgorithm {

 protected:
  Sink<Real> _signal;
  Source<Real> _duration;

  uint64 _nsamples;

 public:
  Duration() : _nsamples(0) {
    declareInputStream(_signal, "signal", "the input signal");
    declareOutputResult(_duration, "duration", "the duration of the signal [s]");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
  }

  void reset();

  void consume();
  void finalProduce();

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_DURATION_H
