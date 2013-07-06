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

#ifndef ESSENTIA_DYNAMICCOMPLEXITY_H
#define ESSENTIA_DYNAMICCOMPLEXITY_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class DynamicComplexity : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<Real> _complexity;
  Output<Real> _loudness;

  int _frameSize;
  Real _sampleRate;

 public:
  DynamicComplexity() {
    declareInput(_signal, "signal", "the input audio signal");
    declareOutput(_complexity, "dynamicComplexity", "the dynamic complexity coefficient");
    declareOutput(_loudness, "loudness", "an estimate of the loudness [dB]");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("frameSize", "the frame size [s]", "(0,inf)", 0.2);
  }


  void configure();
  void compute();

  static const char* name;
  static const char* description;

 protected:
  void filter(std::vector<Real>& result, const std::vector<Real>& input) const;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmcomposite.h"
#include "pool.h"

namespace essentia {
namespace streaming {

class DynamicComplexity : public AlgorithmComposite {

 protected:
  SinkProxy<Real> _signal;
  Source<Real> _complexity;
  Source<Real> _loudness;

  Pool _pool;
  Algorithm* _poolStorage;
  standard::Algorithm* _dynAlgo;

 public:
  DynamicComplexity();
  ~DynamicComplexity() {
    delete _poolStorage;
    delete _dynAlgo;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("frameSize", "the frame size [s]", "(0,inf)", 0.2);
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_poolStorage));
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

#endif // ESSENTIA_DYNAMICCOMPLEXITY_H
