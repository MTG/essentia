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

#ifndef ESSENTIA_REPLAYGAIN_H
#define ESSENTIA_REPLAYGAIN_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class ReplayGain : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<Real> _gain;

  Algorithm* _eqloudFilter;
  int _rmsWindowSize;

 public:
  ReplayGain() {
    declareInput(_signal, "signal", "the input audio signal (must be longer than 0.05ms)");
    declareOutput(_gain, "replayGain", "the distance to the suitable average replay level (~-31dbB) defined by SMPTE [dB]");

    _eqloudFilter = AlgorithmFactory::create("EqualLoudness");
  }

  ~ReplayGain() {
    delete _eqloudFilter;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the input audio signal [Hz]", "(0,inf)", 44100.);
  }

  void configure();

  void reset();

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmcomposite.h"
#include "pool.h"
#include "network.h"

namespace essentia {
namespace streaming {

class ReplayGain : public AlgorithmComposite {

 protected:
  SinkProxy<Real> _signal;
  Source<Real> _gain;

  Algorithm* _eqloud, *_fc, *_instantp;
  scheduler::Network* _network;
  Pool _pool;
  bool _applyEqloud;

 public:
  ReplayGain();
  ~ReplayGain();

  void configure();

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("applyEqloud", "enables whether this algorithm should apply an equal-loudness filter (set to false if the input audio signal is already equal-loudness filtered)", "{true,false}", true);
  }

  void declareProcessOrder() {
    if (_applyEqloud)
      declareProcessStep(ChainFrom(_eqloud));
    else
      declareProcessStep(ChainFrom(_fc));

    declareProcessStep(SingleShot(this));
  }

  AlgorithmStatus process();
  void reset();

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_REPLAYGAIN_H
