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

#ifndef ESSENTIA_STREAMING_MONOMIXER_H
#define ESSENTIA_STREAMING_MONOMIXER_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class MonoMixer : public Algorithm {

 protected:
  Input<int> _channels;
  Input<std::vector<StereoSample> > _inputAudio;
  Output<std::vector<Real> > _outputAudio;

  std::string _type;

 public:
  MonoMixer() {
    declareInput(_inputAudio, "audio", "the input stereo signal");
    declareInput(_channels, "numberChannels", "the number of channels of the input signal");
    declareOutput(_outputAudio, "audio", "the downmixed signal");
  }

  void declareParameters() {
    declareParameter("type", "the type of downmixing performed", "{left,right,mix}", "mix");
  }

  ~MonoMixer() {}

  void configure() {
    _type = parameter("type").toLower();
  }

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia


#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

class MonoMixer : public Algorithm {
 protected:
  Sink<int> _channels;
  Sink<StereoSample> _inputAudio;
  Source<Real> _outputAudio;

  std::string _type;
  int _preferredBufferSize;

 public:
  MonoMixer() : Algorithm() {
    _preferredBufferSize = 4096; // arbitrary
    declareInput(_inputAudio, _preferredBufferSize, "audio", "the input stereo signal");
    declareInput(_channels, "numberChannels", "the number of channels of the input signal");
    declareOutput(_outputAudio, _preferredBufferSize, "audio", "the downmixed signal");

    _outputAudio.setBufferType(BufferUsage::forAudioStream);
  }

  ~MonoMixer() {}

  AlgorithmStatus process();

  void declareParameters() {
    declareParameter("type", "the type of downmixing performed", "{left,right,mix}", "mix");
  }

  void configure() {
    _type = parameter("type").toLower();
  }

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_STREAMING_MONOMIXER_H
