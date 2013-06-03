/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
