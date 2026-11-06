/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_STREAMING_EASYLOADER_H
#define ESSENTIA_STREAMING_EASYLOADER_H


#include "streamingalgorithmcomposite.h"
#include "network.h"

namespace essentia {
namespace streaming {

class EasyLoader : public AlgorithmComposite {
 protected:
  Algorithm* _monoLoader;
  Algorithm* _trimmer;
  Algorithm* _scale;

  SourceProxy<AudioSample> _audio;
  bool _configured;

 public:
  EasyLoader();
  ~EasyLoader();

  void declareParameters() {
    declareParameter("filename", "the name of the file from which to read", "", Parameter::STRING);
    declareParameter("sampleRate", "the output sampling rate [Hz]", "(0,inf)", 44100.);
    declareParameter("startTime", "the start time of the slice to be extracted [s]", "[0,inf)", 0.0);
    declareParameter("endTime", "the end time of the slice to be extracted [s]", "[0,inf)", 1e6);
    declareParameter("replayGain", "the value of the replayGain that should be used to normalize the signal [dB]", "(-inf,inf)", -6.0);
    declareParameter("downmix", "the mixing type for stereo files", "{left,right,mix}", "mix");
    declareParameter("audioStream", "audio stream index to be loaded. Other streams are no taken into account (e.g. if stream 0 is video and 1 is audio use index 0 to access it.)", "[0,inf)", 0);
    declareParameter("gapless", "how to handle the encoder delay and padding that lossy codecs add at the beginning and the end of the decoded signal. \"metadata\" trims exactly the amount the container declares (e.g. the Xing/LAME header of an mp3), which is what a gapless player does and leaves the output sample-aligned with the signal that was encoded. \"decoder\" additionally drops the decoder's own constant latency (529 samples for MPEG Layer III) on streams that declare nothing, removing the decoder's share of the shift but not the encoder's. \"none\" returns the raw decoder output, delay and padding included; 'startTime' and 'endTime' then refer to positions in that raw output, which the decoder reaches by decoding and discarding rather than by seeking.", "{none,metadata,decoder}", "metadata");

  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_monoLoader));
  }

  void configure();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace streaming
} // namespace essentia


#include "vectoroutput.h"
#include "algorithm.h"

namespace essentia {
namespace standard {

// Standard non-streaming algorithm comes after the streaming one as it
// depends on it
class EasyLoader : public Algorithm {
 protected:
  Output<std::vector<AudioSample> > _audio;

  streaming::Algorithm* _loader;
  streaming::VectorOutput<AudioSample>* _audioStorage;
  scheduler::Network* _network;

  void createInnerNetwork();

 public:
  EasyLoader() {
    declareOutput(_audio, "audio", "the audio signal");

    createInnerNetwork();
  }

  ~EasyLoader() {
    delete _network;
  }

  void declareParameters() {
    declareParameter("filename", "the name of the file from which to read", "", Parameter::STRING);
    declareParameter("sampleRate", "the output sampling rate [Hz]", "(0,inf)", 44100.);
    declareParameter("startTime", "the start time of the slice to be extracted [s]", "[0,inf)", 0.0);
    declareParameter("endTime", "the end time of the slice to be extracted [s]", "[0,inf)", 1e6);
    declareParameter("replayGain", "the value of the replayGain that should be used to normalize the signal [dB]", "(-inf,inf)", -6.0);
    declareParameter("downmix", "the mixing type for stereo files", "{left,right,mix}", "mix");
    declareParameter("audioStream", "audio stream index to be loaded. Other streams are no taken into account (e.g. if stream 0 is video and 1 is audio use index 0 to access it.)", "[0,inf)", 0);
    declareParameter("gapless", "how to handle the encoder delay and padding that lossy codecs add at the beginning and the end of the decoded signal. \"metadata\" trims exactly the amount the container declares (e.g. the Xing/LAME header of an mp3), which is what a gapless player does and leaves the output sample-aligned with the signal that was encoded. \"decoder\" additionally drops the decoder's own constant latency (529 samples for MPEG Layer III) on streams that declare nothing, removing the decoder's share of the shift but not the encoder's. \"none\" returns the raw decoder output, delay and padding included; 'startTime' and 'endTime' then refer to positions in that raw output, which the decoder reaches by decoding and discarding rather than by seeking.", "{none,metadata,decoder}", "metadata");

  }

  void configure();

  void compute();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif // ESSENTIA_STREAMING_EASYLOADER_H
