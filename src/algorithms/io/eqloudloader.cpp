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

#include "eqloudloader.h"
#include "algorithmfactory.h"
#include "essentiamath.h"

using namespace std;


namespace essentia {
namespace streaming {

const char* EqloudLoader::name = "EqloudLoader";
const char* EqloudLoader::description = DOC("Given an audio file this algorithm outputs the raw audio data downmixed to mono. Audio is resampled in case the given sampling rate does not match the sampling rate of the input signal and normalized by the given replayGain gain. In addition, audio data is filtered through an equal-loudness filter.\n"
"\n"
"This algorithm uses MonoLoader and thus inherits all of its input requirements and exceptions.\n"
"\n"
"References:\n"
"  [1] Replay Gain - A Proposed Standard,\n"
"  http://replaygain.hydrogenaudio.org\n\n"
"  [2] Replay Gain - Equal Loudness Filter,\n"
"  http://replaygain.hydrogenaudio.org/proposal/equal_loudness.html");

EqloudLoader::EqloudLoader() : AlgorithmComposite(),
                               _monoLoader(0), _trimmer(0), _scale(0), _eqloud(0) {

  declareOutput(_audio, "audio", "the audio signal");

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _monoLoader = factory.create("MonoLoader");
  _trimmer    = factory.create("Trimmer");
  _scale      = factory.create("Scale");
  _eqloud     = factory.create("EqualLoudness");

  _monoLoader->output("audio")  >>  _trimmer->input("signal");
  _trimmer->output("signal")    >>  _scale->input("signal");
  _scale->output("signal")      >>  _eqloud->input("signal");

  attach(_eqloud->output("signal"), _audio);
}

void EqloudLoader::configure() {
  // if no file has been specified, do not do anything
  if (!parameter("filename").isConfigured()) return;

  _monoLoader->configure(INHERIT("filename"),
                         INHERIT("sampleRate"),
                         INHERIT("downmix"));

  _trimmer->configure(INHERIT("sampleRate"),
                      INHERIT("startTime"),
                      INHERIT("endTime"));

  // apply a 6dB preamp, as done by all audio players.
  Real scalingFactor = db2amp(parameter("replayGain").toReal() + 6.0);
  _scale->configure("factor", scalingFactor);

  _eqloud->configure(INHERIT("sampleRate"));
}

} // namespace streaming
} // namespace essentia


namespace essentia {
namespace standard {

const char* EqloudLoader::name = "EqloudLoader";
const char* EqloudLoader::description = DOC("Given an audio file this algorithm outputs the raw audio data downmixed to mono. Audio is resampled in case the given sampling rate does not match the sampling rate of the input signal and normalized by the given replayGain gain. In addition, audio data is filtered through an equal-loudness filter.\n"
"\n"
"This algorithm uses MonoLoader and thus inherits all of its input requirements and exceptions.\n"
"\n"
"References:\n"
"  [1] Replay Gain - A Proposed Standard,\n"
"      http://replaygain.hydrogenaudio.org"
"  [2] Replay Gain - Equal Loudness Filter,\n"
"      http://replaygain.hydrogenaudio.org/equal_loudness.html");


void EqloudLoader::createInnerNetwork() {
  _loader = streaming::AlgorithmFactory::create("EqloudLoader");
  _audioStorage = new streaming::VectorOutput<AudioSample>();

  _loader->output("audio")  >>  _audioStorage->input("data");

  _network = new scheduler::Network(_loader);
}

void EqloudLoader::configure() {
  // if no file has been specified, do not do anything
  // we let the inner loader take care of correctness and sending a nice
  // error message if necessary
  if (!parameter("filename").isConfigured()) return;

  _loader->configure(INHERIT("filename"),
                     INHERIT("sampleRate"),
                     INHERIT("startTime"),
                     INHERIT("endTime"),
                     INHERIT("replayGain"),
                     INHERIT("downmix"));
}

void EqloudLoader::compute() {
  vector<AudioSample>& audio = _audio.get();

  // _audio.reserve(sth_meaningful);

  _audioStorage->setVector(&audio);

  _network->run();
  reset();
}

void EqloudLoader::reset() {
  _network->reset();
}

} // namespace standard
} // namespace essentia
