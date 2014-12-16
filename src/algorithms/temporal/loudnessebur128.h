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

#ifndef ESSENTIA_LOUDNESSEBUR128_H
#define ESSENTIA_LOUDNESSEBUR128_H

#include "algorithmfactory.h"
#include "network.h"
#include "pool.h"
#include "streamingalgorithmcomposite.h"

namespace essentia {
namespace streaming {

class LoudnessEBUR128 : public AlgorithmComposite {

 protected:
  Algorithm* _loudnessEBUR128Filter;
  Algorithm* _frameCutterMomentary;
  Algorithm* _frameCutterShortTerm;
  Algorithm* _frameCutterIntegrated;
  Algorithm* _meanMomentary;
  Algorithm* _meanShortTerm;
  Algorithm* _meanIntegrated;
  Algorithm* _computeMomentary;
  Algorithm* _computeShortTerm;
  Algorithm* _computeIntegrated;

  SinkProxy<StereoSample> _signal;
  SourceProxy<Real> _momentaryLoudness;
  SourceProxy<Real> _shortTermLoudness;
  Source<Real> _integratedLoudness;
  Source<Real> _loudnessRange;
  SourceProxy<Real> _momentaryLoudnessMax;
  SourceProxy<Real> _shortTermLoudnessMax;

  Pool _pool;

  scheduler::Network* _network;

  int _hopSize;

 public:
  LoudnessEBUR128();
   ~LoudnessEBUR128();

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_loudnessEBUR128Filter));
    declareProcessStep(SingleShot(this));
  }

  void declareParameters() {
    // pre-processing
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    // specs: the update rate for short-term loudness ‘live meters’ shall be at least 10 Hz
    declareParameter("hopSize", "the hop size with which the loudness is computed [s]", "(0,0.1]", 0.1);  
  };

  void configure();
  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_LOUDNESSEBUR128_H
