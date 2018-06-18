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

#ifndef ESSENTIA_STREAMING_STEREOMUXER_H
#define ESSENTIA_STREAMING_STEREOMUXER_H

#include "streamingalgorithm.h"
#include "network.h"

namespace essentia {
namespace streaming {

class StereoMuxer : public Algorithm {
 protected:
  Sink<AudioSample> _left;
  Sink<AudioSample> _right;
  Source<StereoSample> _audio;

  int _preferredBufferSize;

 public:
  StereoMuxer() : Algorithm() {
    _preferredBufferSize = 4096; // arbitrary
    declareInput(_left, _preferredBufferSize, "left", "the left channel of the audio signal");
    declareInput(_right, _preferredBufferSize, "right", "the right channel of the audio signal");
    declareOutput(_audio, _preferredBufferSize, "audio", "the output stereo signal");

    _audio.setBufferType(BufferUsage::forAudioStream);
  }

  ~StereoMuxer() {}

  AlgorithmStatus process();

  void declareParameters() {}

  void configure() {}

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#include "algorithm.h"

namespace essentia {
namespace standard {

class StereoMuxer : public Algorithm {
 protected:
  Input<std::vector<AudioSample> > _left;
  Input<std::vector<AudioSample> > _right;
  Output<std::vector<StereoSample> > _audio;

 public:
  StereoMuxer() {
    declareInput(_left, "left", "the left channel of the audio signal");
    declareInput(_right, "right", "the right channel of the audio signal");
    declareOutput(_audio, "audio", "the audio signal");
  }

  void declareParameters() {}
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif // ESSENTIA_STREAMING_STEREOMUXER_H
