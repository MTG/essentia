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

#ifndef ESSENTIA_STREAMING_STEREODEMUXER_H
#define ESSENTIA_STREAMING_STEREODEMUXER_H

#include "streamingalgorithm.h"
#include "network.h"

namespace essentia {
namespace streaming {

class StereoDemuxer : public Algorithm {
 protected:
  Sink<StereoSample> _audio;
  Source<AudioSample> _left;
  Source<AudioSample> _right;

  int _preferredBufferSize;

 public:
  StereoDemuxer() : Algorithm() {
    _preferredBufferSize = 4096; // arbitrary
    declareInput(_audio, _preferredBufferSize, "audio", "the input stereo signal");
    declareOutput(_left, _preferredBufferSize, "left", "the left channel of the audio signal");
    declareOutput(_right, _preferredBufferSize, "right", "the right channel of the audio signal");

    _left.setBufferType(BufferUsage::forAudioStream);
    _right.setBufferType(BufferUsage::forAudioStream);
  }

  ~StereoDemuxer() {}

  AlgorithmStatus process();

  void declareParameters() {}

  void configure() {}

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#include "algorithm.h"
#include "network.h"
#include "vectoroutput.h"
#include "vectorinput.h"

namespace essentia {
namespace standard {

// Standard non-streaming algorithm comes after the streaming one as it
// depends on it
class StereoDemuxer : public Algorithm {
 protected:
  Input<std::vector<StereoSample> > _audio;
  Output<std::vector<AudioSample> > _left;
  Output<std::vector<AudioSample> > _right;

  streaming::Algorithm* _demuxer;
  streaming::VectorInput<StereoSample, 4096>* _audiogen;
  streaming::VectorOutput<AudioSample>* _leftStorage;
  streaming::VectorOutput<AudioSample>* _rightStorage;
  scheduler::Network* _network;

  void createInnerNetwork();

 public:
  StereoDemuxer() {
    declareInput(_audio, "audio", "the audio signal");
    declareOutput(_left, "left", "the left channel of the audio signal");
    declareOutput(_right, "right", "the right channel of the audio signal");

    createInnerNetwork();
  }

  ~StereoDemuxer() {
    delete _network;
  }

  void declareParameters() {}

  void configure(){};
  void reset() {
    _network->reset();
  }

  void compute();

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif // ESSENTIA_STREAMING_STEREODEMUXER_H
