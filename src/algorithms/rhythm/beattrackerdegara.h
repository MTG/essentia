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

#ifndef BEATTRACKERDEGARA_H
#define BEATTRACKERDEGARA_H

#include "streamingalgorithmcomposite.h"
#include "pool.h"
#include "algorithm.h"
#include "network.h"

namespace essentia {
namespace streaming {

class BeatTrackerDegara : public AlgorithmComposite {

 protected:
  SinkProxy<Real>_signal;
  SourceProxy<Real> _ticks;

  Pool _pool;

  // algorithm numeration corresponds to the process chains
  Algorithm* _frameCutter;
  Algorithm* _windowing;
  Algorithm* _fft;
  Algorithm* _cart2polar;
  Algorithm* _onsetComplex;
  Algorithm* _ticksComplex;

  scheduler::Network* _network;
  bool _configured;

  void createInnerNetwork();
  void clearAlgos();
  Real _sampleRate;

 public:
  BeatTrackerDegara();
  ~BeatTrackerDegara();

  void declareParameters() {
    //declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("maxTempo", "the fastest tempo to detect [bpm]", "[60,250]", 208);
    declareParameter("minTempo", "the slowest tempo to detect [bpm]", "[40,180]", 40);
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_frameCutter));
  }

  void configure();
  void reset();

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#include "vectorinput.h"

namespace essentia {
namespace standard {

class BeatTrackerDegara : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _ticks;

  streaming::Algorithm* _beatTracker;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

 public:

  BeatTrackerDegara();
  ~BeatTrackerDegara();

  void declareParameters() {
    //declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("maxTempo", "the fastest tempo to detect [bpm]", "[60,250]", 208);
    declareParameter("minTempo", "the slowest tempo to detect [bpm]", "[40,180]", 40);
  }

  void configure();
  void compute();
  void reset();
  void createInnerNetwork();

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif // BEATTRACKERDEGARA_H
