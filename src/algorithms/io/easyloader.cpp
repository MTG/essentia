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

#include "easyloader.h"
#include "algorithmfactory.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* EasyLoader::name = essentia::standard::EasyLoader::name;
const char* EasyLoader::category = essentia::standard::EasyLoader::category;
const char* EasyLoader::description = essentia::standard::EasyLoader::description;


EasyLoader::EasyLoader() : AlgorithmComposite(),
                           _monoLoader(0), _trimmer(0), _scale(0), _configured(false) {

  declareOutput(_audio, "audio", "the output audio signal");

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _monoLoader = factory.create("MonoLoader");
  _trimmer    = factory.create("Trimmer");
  _scale      = factory.create("Scale");

  _monoLoader->output("audio")  >>  _trimmer->input("signal");
  _trimmer->output("signal")    >>  _scale->input("signal");

  attach(_scale->output("signal"), _audio);
}

EasyLoader::~EasyLoader() {
  delete _monoLoader;
  delete _trimmer;
  delete _scale;
}

void EasyLoader::configure() {
  // if no file has been specified, do not do anything
  if (!parameter("filename").isConfigured()) return;

  _monoLoader->configure(INHERIT("filename"),
                         INHERIT("sampleRate"),
                         INHERIT("downmix"),
                         INHERIT("audioStream"),
                         INHERIT("gapless"));

  Parameter originalSampleRate = _monoLoader->parameter("originalSampleRate");
  _params.add("originalSampleRate", originalSampleRate);

  // Issue #771: startTime/endTime used to be applied by the Trimmer AFTER the whole stream had
  // been decoded, so a 10 s slice of a 2 h recording cost the whole 2 h. Hand them to the
  // loader instead -- it seeks to startTime and stops at endTime, making the cost proportional
  // to the slice. The parameters keep exactly their meaning, and they select exactly the same
  // samples: the loader converts seconds to samples with Trimmer's own truncation rule.
  //
  // ONE CASE STAYS ON THE OLD PATH, and it is not laziness: when the output rate differs from
  // the file's rate, libsamplerate's output depends on how much input it has already consumed
  // (filter history plus a phase accumulator). A converter started at startTime therefore does
  // not produce the samples a converter started at 0 produces -- measured at about -49 dB
  // relative RMS, uniformly across the slice, and NOT removable by any bounded amount of
  // preroll. Seeking here would silently change what EasyLoader returns for every existing
  // caller that resamples, so we do not. Callers that want the win and can accept that residual
  // have MonoLoader's own startTime/endTime.
  // Equality of the two rates is exactly Resample's own `src_ratio == 1.0` short circuit, i.e.
  // precisely the condition under which the converter is a fastcopy and carries no state.
  if (originalSampleRate.toReal() == parameter("sampleRate").toReal()) {
    _monoLoader->configure(INHERIT("filename"),
                           INHERIT("sampleRate"),
                           INHERIT("downmix"),
                           INHERIT("audioStream"),
                           INHERIT("startTime"),
                           INHERIT("endTime"),
                           INHERIT("gapless"));

    // The loader already delivered exactly the requested slice.
    _trimmer->configure("sampleRate", parameter("sampleRate"),
                        "startTime", 0.0,
                        "endTime", 1.0e6);
  }
  else {
    _trimmer->configure(INHERIT("sampleRate"),
                        INHERIT("startTime"),
                        INHERIT("endTime"));
  }

  // apply a 6dB preamp, as done by all audio players.
  Real scalingFactor = db2amp(parameter("replayGain").toReal() + 6.0);

  _scale->configure("factor", scalingFactor);
}

} // namespace streaming
} // namespace essentia


namespace essentia {
namespace standard {

const char* EasyLoader::name = "EasyLoader";
const char* EasyLoader::category = "Input/output";
const char* EasyLoader::description = DOC("This algorithm loads the raw audio data from an audio file, downmixes it to mono and normalizes using replayGain. The audio is resampled in case the given sampling rate does not match the sampling rate of the input signal and is normalized by the given replayGain value.\n"
"\n"
"This algorithm uses MonoLoader and therefore inherits all of its input requirements and exceptions.\n"
"\n"
"The 'gapless' parameter is handed to AudioLoader and decides whether the encoder delay "
"and padding that lossy codecs add at both ends of the signal are trimmed; the default "
"trims exactly what the file declares. See AudioLoader for what each value means."
"\n"
"\n"
"References:\n"
"  [1] Replay Gain - A Proposed Standard,\n"
"  http://replaygain.hydrogenaudio.org");


void EasyLoader::createInnerNetwork() {
  _loader = streaming::AlgorithmFactory::create("EasyLoader");
  _audioStorage = new streaming::VectorOutput<AudioSample>();

  _loader->output("audio")  >>  _audioStorage->input("data");

  _network = new scheduler::Network(_loader);
}

void EasyLoader::configure() {
  // if no file has been specified, do not do anything
  // we let the inner loader take care of correctness and sending a nice
  // error message if necessary
  if (!parameter("filename").isConfigured()) return;

  _loader->configure(INHERIT("filename"),
                     INHERIT("sampleRate"),
                     INHERIT("startTime"),
                     INHERIT("endTime"),
                     INHERIT("replayGain"),
                     INHERIT("downmix"),
                     INHERIT("audioStream"),
                     INHERIT("gapless"));
}

void EasyLoader::compute() {
  vector<AudioSample>& audio = _audio.get();
  audio.clear();
  // TODO: somehow retrieve the audioFileLength from the internal loader at
  // configure time
  // _audio.reserve( sampleRate*( min(endTime, audioFileLength) - startTime ) );

  _audioStorage->setVector(&audio);

  _network->run();
  reset();
}

void EasyLoader::reset() {
  _network->reset();
}

} // namespace standard
} // namespace essentia
