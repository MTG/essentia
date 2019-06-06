/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_LOUDNESSEBUR128FILTER_H
#define ESSENTIA_LOUDNESSEBUR128FILTER_H

#include "algorithmfactory.h"
#include "network.h"
#include "streamingalgorithmcomposite.h"

namespace essentia {
namespace streaming {

class LoudnessEBUR128Filter : public AlgorithmComposite {

 protected:
  Algorithm* _stereoDemuxer;
  Algorithm* _filterLeft;
  Algorithm* _filterRight; 
  Algorithm* _squareLeft;
  Algorithm* _squareRight;
  Algorithm* _sum;
  
  SinkProxy<StereoSample> _signal;
  SourceProxy<Real> _signalFiltered;

  scheduler::Network* _network;

 public:
  LoudnessEBUR128Filter();
  ~LoudnessEBUR128Filter();

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_stereoDemuxer));
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
  };

  void configure();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_LOUDNESSEBUR128FILTER_H
