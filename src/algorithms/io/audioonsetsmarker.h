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

#ifndef ESSENTIA_AUDIOONSETSMARKER_H
#define ESSENTIA_AUDIOONSETSMARKER_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class AudioOnsetsMarker : public Algorithm {

 protected:
  Input<std::vector<Real> > _input;
  Output<std::vector<Real> > _output;

  Real _sampleRate;
  std::vector<Real> _onsets;

  bool _beep;

 public:
  AudioOnsetsMarker() : _beep(false) {
    declareInput(_input, "signal", "the input signal");
    declareOutput(_output, "signal", "the input signal mixed with bursts at onset locations");
  }

  ~AudioOnsetsMarker() {}

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the output signal [Hz]", "(0,inf)", 44100.);
    declareParameter("type", "the type of sound to be added on the event", "{beep,noise}", "beep");
    declareParameter("onsets", "the list of onset locations [s]", "", std::vector<Real>());
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia


#include "streamingalgorithm.h"
#include "network.h"

namespace essentia {
namespace streaming {

class AudioOnsetsMarker : public Algorithm {
 protected:
  Sink<Real> _input;
  Source<Real> _output;

  Real _sampleRate;
  std::vector<Real> _burst;
  std::vector<Real> _onsets;
  bool _beep;
  int _onsetIdx, _burstIdx;
  int _processedSamples;
  int _preferredSize;

 public:
  AudioOnsetsMarker();
  ~AudioOnsetsMarker() {}

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the output signal [Hz]", "(0,inf)", 44100.);
    declareParameter("type", "the type of sound to be added on the event", "{beep,noise}", "beep");
    declareParameter("onsets", "the list of onset locations [s]", "", std::vector<Real>());
  }

  void createInnerNetwork();
  void configure();
  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_AUDIOONSETSMARKER_H
