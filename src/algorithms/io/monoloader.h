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

#ifndef ESSENTIA_STREAMING_MONOLOADER_H
#define ESSENTIA_STREAMING_MONOLOADER_H


#include "streamingalgorithmcomposite.h"
#include "network.h"

namespace essentia {
namespace streaming {

class MonoLoader : public AlgorithmComposite {
 protected:
  Algorithm* _audioLoader;
  Algorithm* _mixer;
  Algorithm* _resample;

  SourceProxy<AudioSample> _audio;
  bool _configured;

 public:
  MonoLoader();

  ~MonoLoader() {
    // Disconnect all null connections to delete the corresponding DevNull objects created.
    disconnect(_audioLoader->output("md5"), NOWHERE);
    disconnect(_audioLoader->output("bit_rate"), NOWHERE);
    disconnect(_audioLoader->output("codec"), NOWHERE);
    disconnect(_audioLoader->output("sampleRate"), NOWHERE);
    
    delete _audioLoader;
    delete _mixer;
    delete _resample;
  }

  void declareParameters() {
    declareParameter("filename", "the name of the file from which to read", "", Parameter::STRING);
    declareParameter("sampleRate", "the desired output sampling rate [Hz]", "(0,inf)", 44100.);
    declareParameter("downmix", "the mixing type for stereo files", "{left,right,mix}", "mix");
    declareParameter("audioStream", "audio stream index to be loaded. Other streams are no taken into account (e.g. if stream 0 is video and 1 is audio use index 0 to access it.)", "[0,inf)", 0);

  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_audioLoader));
  }

  void configure();

  static const char* name;
  static const char* category;
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
class MonoLoader : public Algorithm {
 protected:
  Output<std::vector<AudioSample> > _audio;

  streaming::Algorithm* _loader;
  streaming::VectorOutput<AudioSample>* _audioStorage;
  scheduler::Network* _network;

  void createInnerNetwork();

 public:
  MonoLoader() {
    declareOutput(_audio, "audio", "the audio signal");

    createInnerNetwork();
  }

  ~MonoLoader() {
    delete _network;
  }

  void declareParameters() {
    declareParameter("filename", "the name of the file from which to read", "", Parameter::STRING);
    declareParameter("sampleRate", "the desired output sampling rate [Hz]", "(0,inf)", 44100.);
    declareParameter("downmix", "the mixing type for stereo files", "{left,right,mix}", "mix");
    declareParameter("audioStream", "audio stream index to be loaded. Other streams are no taken into account (e.g. if stream 0 is video and 1 is audio use index 0 to access it.)", "[0,inf)", 0);

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


#endif // ESSENTIA_STREAMING_MONOLOADER_H
