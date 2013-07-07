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

#ifndef ESSENTIA_BARK_EXTRACTOR_H
#define ESSENTIA_BARK_EXTRACTOR_H

#include "streamingalgorithmcomposite.h"
#include "pool.h"
#include "algorithm.h"
#include "network.h"

namespace essentia {
namespace streaming {

class BarkExtractor : public AlgorithmComposite {
 protected:
  SinkProxy<Real> _signal;

  SourceProxy<std::vector<Real> > _bbands;
  SourceProxy<Real> _bbandsKurtosis;
  SourceProxy<Real> _bbandsSkewness;
  SourceProxy<Real> _bbandsSpread;
  SourceProxy<Real> _flatness;
  SourceProxy<Real> _crestValue;

  Algorithm *_barkBands, *_centralMoments, *_crest,
            *_distributionShape, *_flatnessdb,
            *_frameCutter, *_spectrum, *_windowing;

  scheduler::Network* _network;

  bool _configured;
  void clearAlgos();

 public:
  BarkExtractor();
  ~BarkExtractor();

  void declareParameters() {
    declareParameter("frameSize", "the frame size for computing low level features", "(0,inf)", 2048);
    declareParameter("hopSize", "the hop size for computing low level features", "(0,inf)", 1024);
    declareParameter("sampleRate", "the audio sampling rate", "(0,inf)", 44100.0);
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


#endif // ESSENTIA_BARK_EXTRACTOR_H
