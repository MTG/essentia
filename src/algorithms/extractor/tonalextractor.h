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

#ifndef TONAL_EXTRACTOR_H
#define TONAL_EXTRACTOR_H

#include "streamingalgorithmcomposite.h"
#include "pool.h"
#include "algorithm.h"
#include "vectorinput.h"
#include "network.h"

namespace essentia {
namespace streaming {

// FIXME: check this actually works, the connections are probably of the wrong type and everything most likely broken
class TonalExtractor : public AlgorithmComposite {
 protected:
  SinkProxy<Real> _signal;

  SourceProxy<Real> _chordsChangesRate;
  SourceProxy<std::vector<Real> > _chordsHistogram;
  SourceProxy<std::string> _chordsKey;
  SourceProxy<Real> _chordsNumberRate;
  SourceProxy<std::string> _chordsProgression;
  SourceProxy<std::string> _chordsScale;
  SourceProxy<Real> _chordsStrength;
  SourceProxy<std::vector<Real> > _hpcps;
  SourceProxy<std::vector<Real> > _hpcpsTuning;
  SourceProxy<std::string> _keyKey;
  SourceProxy<std::string> _keyScale;
  SourceProxy<Real> _keyStrength;

  Algorithm *_frameCutter, *_windowing, *_spectrum, *_spectralPeaks,
            *_hpcpKey, *_hpcpChord, *_hpcpTuning, *_key,
            *_chordsDescriptors, *_chordsDetection;

  scheduler::Network* _network;

 public:
  TonalExtractor();
  ~TonalExtractor();

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

class TonalExtractor : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _chordsHistogram;
  Output<Real> _chordsChangesRate;
  Output<std::string> _chordsKey;
  Output<Real> _chordsNumberRate;
  Output<std::vector<std::string> > _chords;
  Output<std::string> _chordsScale;
  Output<std::vector<Real> > _chordsStrength;
  Output<std::vector<std::vector<Real> > > _hpcp;
  Output<std::vector<std::vector<Real> > > _hpcpHighRes;
  Output<std::string> _key;
  Output<std::string> _scale;
  Output<Real> _keyStrength;

  streaming::Algorithm* _tonalExtractor;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

 public:

  TonalExtractor();
  ~TonalExtractor();

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

#endif
