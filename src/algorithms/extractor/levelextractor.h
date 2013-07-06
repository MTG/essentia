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

#ifndef LEVEL_EXTRACTOR_H
#define LEVEL_EXTRACTOR_H

#include "streamingalgorithmcomposite.h"
#include "pool.h"
#include "algorithm.h"
#include "vectorinput.h"
#include "network.h"

namespace essentia {
namespace streaming {


class LevelExtractor : public AlgorithmComposite {
 protected:
  SinkProxy<Real> _signal;
  SourceProxy<Real> _loudnessValue;

  Algorithm* _frameCutter;
  Algorithm* _loudness;

 public:
  LevelExtractor();
  ~LevelExtractor();

  void declareParameters() {
    declareParameter("frameSize", "frame size to compute loudness", "(0,inf)", 88200);
    declareParameter("hopSize", "hop size to compute loudness", "(0,inf)", 44100);
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_frameCutter));
  }

  void configure();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia


namespace essentia {
namespace standard {

class LevelExtractor : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _loudness;

  bool _configured;

  streaming::Algorithm* _levelExtractor;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

 public:

  LevelExtractor();
  ~LevelExtractor();

  void declareParameters() {
    declareParameter("frameSize", "frame size to compute loudness", "(0,inf)", 88200);
    declareParameter("hopSize", "hop size to compute loudness", "(0,inf)", 44100);
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


#endif // LEVEL_EXTRACTOR_H
