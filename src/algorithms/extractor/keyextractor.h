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

#ifndef KEY_EXTRACTOR_H
#define KEY_EXTRACTOR_H

#include "streamingalgorithmcomposite.h"
#include "pool.h"
#include "algorithm.h"
#include "vectorinput.h"
#include "network.h"

namespace essentia {
namespace streaming {

class KeyExtractor : public AlgorithmComposite {
 protected:
  Algorithm *_frameCutter, *_windowing, *_spectrum, *_spectralPeaks, *_hpcpKey, *_key;
  scheduler::Network* _network;
  bool _configured;

  SinkProxy<Real> _audio;
  SourceProxy<std::string> _keyKey;
  SourceProxy<std::string> _keyScale;
  SourceProxy<Real> _keyStrength;

 public:
  KeyExtractor();
  ~KeyExtractor();

  void declareParameters() {
    declareParameter("frameSize", "the framesize for computing tonal features", "(0,inf)", 4096);
    declareParameter("hopSize", "the hopsize for computing tonal features", "(0,inf)", 2048);
    declareParameter("tuningFrequency", "the tuning frequency of the input signal", "(0,inf)", 440.0);
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

class KeyExtractor : public Algorithm {
 protected:
  Input<std::vector<Real> > _audio;
  Output<std::string> _key;
  Output<std::string> _scale;
  Output<Real> _strength;

  bool _configured;

  streaming::Algorithm* _keyExtractor;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

 public:

  KeyExtractor();
  ~KeyExtractor();

  void declareParameters() {
    declareParameter("frameSize", "the framesize for computing tonal features", "(0,inf)", 4096);
    declareParameter("hopSize", "the hopsize for computing tonal features", "(0,inf)", 2048);
    declareParameter("tuningFrequency", "the tuning frequency of the input signal", "(0,inf)", 440.0);
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

#endif // KEY_EXTRACTOR_H
