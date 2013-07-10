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

#ifndef TUNING_FREQUENCY_EXTRACTOR_H
#define TUNING_FREQUENCY_EXTRACTOR_H

#include "streamingalgorithmcomposite.h"
#include "algorithm.h"
#include "pool.h"
#include "vectorinput.h"
#include "network.h"

namespace essentia {
namespace streaming {

class TuningFrequencyExtractor : public AlgorithmComposite {
 protected:
  Algorithm* _frameCutter, *_spectralPeaks, *_spectrum, *_tuningFrequency, *_windowing;

  SinkProxy<Real> _signal;
  SourceProxy<Real> _tuningFreq;

 public:
  TuningFrequencyExtractor();
  ~TuningFrequencyExtractor();

  void declareParameters() {
    declareParameter("frameSize", "the frameSize for computing tuning frequency", "(0,inf)", 4096);
    declareParameter("hopSize", "the hopsize for computing tuning frequency", "(0,inf)", 2048);
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_frameCutter));
  }

  void configure();
  void createInnerNetwork();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia


namespace essentia {
namespace standard {

class TuningFrequencyExtractor : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _tuningFrequency;

  bool _configured;

  streaming::Algorithm* _tuningFrequencyExtractor;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

 public:

  TuningFrequencyExtractor();
  ~TuningFrequencyExtractor();

  void declareParameters() {
    declareParameter("frameSize", "the frameSize for computing tuning frequency", "(0,inf)", 4096);
    declareParameter("hopSize", "the hopsize for computing tuning frequency", "(0,inf)", 2048);
  }

  void configure();
  void createInnerNetwork();
  void compute();
  void reset();

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif
