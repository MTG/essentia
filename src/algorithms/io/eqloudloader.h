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

#ifndef ESSENTIA_STREAMING_EQLOUDLOADER_H
#define ESSENTIA_STREAMING_EQLOUDLOADER_H


#include "streamingalgorithmcomposite.h"
#include "network.h"

namespace essentia {
namespace streaming {

class EqloudLoader : public AlgorithmComposite {
 protected:
  Algorithm* _monoLoader;
  Algorithm* _trimmer;
  Algorithm* _scale;
  Algorithm* _eqloud;

  SourceProxy<AudioSample> _audio;

 public:
  EqloudLoader();

  ~EqloudLoader() {
    delete _monoLoader;
    delete _trimmer;
    delete _scale;
    delete _eqloud;
  }

  void declareParameters() {
    declareParameter("filename", "the name of the file from which to read", "", Parameter::STRING);
    declareParameter("sampleRate", "the output sampling rate [Hz]", "{32000,44100,48000}", 44100.);
    declareParameter("startTime", "the start time of the slice to be extracted [s]", "[0,inf)", 0.0);
    declareParameter("endTime", "the end time of the slice to be extracted [s]", "[0,inf)", 1e6);
    declareParameter("replayGain", "the value of the replayGain [dB] that should be used to normalize the signal [dB]", "(-inf,inf)", -6.0);
    declareParameter("downmix", "the mixing type for stereo files", "{left,right,mix}", "mix");
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_monoLoader));
  }

  void configure();

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia


#include "vectoroutput.h"
#include "network.h"
#include "algorithm.h"

namespace essentia {
namespace standard {

// Standard non-streaming algorithm comes after the streaming one as it
// depends on it
class EqloudLoader : public Algorithm {
 protected:
  Output<std::vector<AudioSample> > _audio;

  streaming::Algorithm* _loader;
  streaming::VectorOutput<AudioSample>* _audioStorage;
  scheduler::Network* _network;

  void createInnerNetwork();

 public:
  EqloudLoader() {
    declareOutput(_audio, "audio", "the audio signal");

    createInnerNetwork();
  }

  ~EqloudLoader() {
    delete _network;
  }

  void declareParameters() {
    declareParameter("filename", "the name of the file from which to read", "", Parameter::STRING);
    declareParameter("sampleRate", "the output sampling rate [Hz]", "{32000,44100,48000}", 44100.);
    declareParameter("startTime", "the start time of the slice to be extracted [s]", "[0,inf)", 0.0);
    declareParameter("endTime", "the end time of the slice to be extracted [s]", "[0,inf)", 1e6);
    declareParameter("replayGain", "the value of the replayGain [dB] that should be used to normalize the signal [dB]", "(-inf,inf)", -6.0);
    declareParameter("downmix", "the mixing type for stereo files", "{left,right,mix}", "mix");
  }

  void configure();

  void compute();
  void reset();

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia


#endif // ESSENTIA_STREAMING_EQLOUDLOADER_H
