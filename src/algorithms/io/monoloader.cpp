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

#include "monoloader.h"
#include "algorithmfactory.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* MonoLoader::name = "MonoLoader";
const char* MonoLoader::description = DOC("Given an audio file this algorithm outputs the raw audio data downmixed to mono. Audio is resampled in case the given sampling rate does not match the sampling rate of the input signal.\n"
"\n"
"This algorithm uses AudioLoader and thus inherits all of its input requirements and exceptions.");


MonoLoader::MonoLoader() : AlgorithmComposite(),
                           _audioLoader(0), _mixer(0), _resample(0), _configured(false) {

  declareOutput(_audio, "audio", "the mono audio signal");

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _audioLoader = factory.create("AudioLoader");
  _mixer       = factory.create("MonoMixer");
  _resample    = factory.create("Resample");

  _audioLoader->output("audio")           >>  _mixer->input("audio");
  _audioLoader->output("numberChannels")  >>  _mixer->input("numberChannels");
  _mixer->output("audio")                 >>  _resample->input("signal");

  attach(_resample->output("signal"), _audio);
}

void MonoLoader::configure() {
  Parameter filename = parameter("filename");
  // if no file has been specified, do not do anything
  if (!filename.isConfigured()) return;

  _audioLoader->configure("filename", filename);

  int inputSampleRate = (int)lastTokenProduced<Real>(_audioLoader->output("sampleRate"));

  // TODO: this should probably be turned into a source as well, same as what's done above for audioLoader->sampleRate
  // also keep it as a parameter (ugly), but act as an optional source (no need
  // to connect, etc...)
  _params.add("originalSampleRate", inputSampleRate);

  _resample->configure("inputSampleRate", inputSampleRate,
                       "outputSampleRate", parameter("sampleRate"));

  _mixer->configure("type", parameter("downmix"));

}

} // namespace streaming
} // namespace essentia



namespace essentia {
namespace standard {

const char* MonoLoader::name = "MonoLoader";
const char* MonoLoader::description = DOC("Given an audio file this algorithm outputs the raw audio data downmixed to mono. Audio is resampled in case the given sampling rate does not match the sampling rate of the input signal.\n"
"\n"
"This algorithm uses AudioLoader and thus inherits all of its input requirements and exceptions.");


void MonoLoader::createInnerNetwork() {
  _loader = streaming::AlgorithmFactory::create("MonoLoader");
  _audioStorage = new streaming::VectorOutput<AudioSample>();

  connect(_loader->output("audio"), _audioStorage->input("data"));

  _network = new scheduler::Network(_loader);
}

void MonoLoader::configure() {
  // if no file has been specified, do not do anything
  if (!parameter("filename").isConfigured()) return;

  _loader->configure(INHERIT("filename"),
                     INHERIT("sampleRate"),
                     INHERIT("downmix"));
}

void MonoLoader::compute() {
  vector<AudioSample>& audio = _audio.get();

  // TODO: _audio.reserve(sth_meaningful);

  _audioStorage->setVector(&audio);

  _network->run();
  reset();
}

void MonoLoader::reset() {
  _network->reset();
}

} // namespace standard
} // namespace essentia
