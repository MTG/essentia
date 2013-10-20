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

#ifndef BEATTRACKERMULTIFEATURE_H
#define BEATTRACKERMULTIFEATURE_H

#include "streamingalgorithmcomposite.h"
#include "pool.h"
#include "algorithm.h"
#include "network.h"

namespace essentia {
namespace streaming {

class BeatTrackerMultiFeature : public AlgorithmComposite {

 protected:
  SinkProxy<Real>_signal;
  Source<Real> _ticks;
  Source<Real> _confidence;

  Pool _pool;

  // algorithm numeration corresponds to the process chains
  Algorithm* _frameCutter1;
  Algorithm* _windowing1;
  Algorithm* _fft1;
  Algorithm* _cart2polar1;
  Algorithm* _onsetRms1;
  Algorithm* _onsetComplex1;
  Algorithm* _ticksRms1;
  Algorithm* _ticksComplex1;
  Algorithm* _onsetMelFlux1;
  Algorithm* _ticksMelFlux1;

  Algorithm* _onsetBeatEmphasis3;
  Algorithm* _ticksBeatEmphasis3;

  Algorithm* _onsetInfogain4;
  Algorithm* _ticksInfogain4;

  standard::Algorithm* _tempoTapMaxAgreement;

  Algorithm* _scale;

  scheduler::Network* _network;
  bool _configured;

  void createInnerNetwork();
  void clearAlgos();
  Real _sampleRate;

 public:
  BeatTrackerMultiFeature();

  ~BeatTrackerMultiFeature();

  void declareParameters() {
    //declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("maxTempo", "the fastest tempo to detect [bpm]", "[60,250]", 208);
    declareParameter("minTempo", "the slowest tempo to detect [bpm]", "[40,180]", 40);
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_scale));
    declareProcessStep(SingleShot(this));
  }

  void configure();
  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#include "vectorinput.h"

namespace essentia {
namespace standard {

class BeatTrackerMultiFeature : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _ticks;
  Output<Real> _confidence;

  streaming::Algorithm* _beatTracker;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

 public:

  BeatTrackerMultiFeature();
  ~BeatTrackerMultiFeature();

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

#endif // BEATTRACKERMULTIFEATURE_H
