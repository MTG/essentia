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

#include "audiowriter.h"

extern "C" {
#include <libavformat/avformat.h>
}

using namespace std;


namespace essentia {
namespace streaming {

const char* AudioWriter::name = "AudioWriter";
const char* AudioWriter::description = DOC("This algorithm encodes an input signal into a stereo audio file.\n\n"

"Supported formats are wav, aiff, mp3, flac and ogg.\n\n"

"An exception is thrown when other extensions are given. Note that to encode in mp3 format it is mandatory that ffmpeg was configured with mp3 enabled.");


void AudioWriter::configure() {

  if (!parameter("filename").isConfigured() || parameter("filename").toString().empty()) {
    // no file has been specified or retarded name specified, do nothing
    _configured = false;
    return;
  }

  reset();
  _configured = true;
}

AlgorithmStatus AudioWriter::process() {
  if (!_configured) {
    throw EssentiaException("AudioWriter: Trying to call process() on an AudioWriter algo which hasn't been correctly configured");
  }

  if (!_audioCtx.isOpen()) _audioCtx.open();

  EXEC_DEBUG("process()");
  AlgorithmStatus status = acquireData();

  if (status != OK) {
    if (!shouldStop()) return status;

    // encode whatever is left over
    int available = _audio.available();

    if (available == 0) {
      EXEC_DEBUG("End of stream. There are 0 available tokens");
      shouldStop(true);
      _audioCtx.close();
      return FINISHED;
    }

    EXEC_DEBUG("Audio frame could not be fully acquired.");
    EXEC_DEBUG("There are " << available << " available tokens");
    _audio.setAcquireSize(available);
    _audio.setReleaseSize(available);

    return process();
  }

  try {
    _audioCtx.write(_audio.tokens());
  }
  catch (EssentiaException& e) {
    throw EssentiaException("AudioWriter: error writing to audio file: ", e.what());
  }

  releaseData();

  return OK;
}

void AudioWriter::reset() {
  Algorithm::reset();

  int recommendedBufferSize;
  try {
    recommendedBufferSize = _audioCtx.create(parameter("filename").toString(),
                                             parameter("format").toString(),
                                             2, // nChannels
                                             parameter("sampleRate").toInt(),
                                             parameter("bitrate").toInt()*1000);
  }
  catch (EssentiaException& e) {
    throw EssentiaException("AudioWriter: Error creating audio file: ", e.what());
  }

  _audio.setAcquireSize(recommendedBufferSize);
  _audio.setReleaseSize(recommendedBufferSize);
}


} // namespace streaming
} // namespace essentia

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

const char* AudioWriter::name = "AudioWriter";
const char* AudioWriter::description = DOC("This algorithm encodes an input signal into a stereo audio file.\n\n"

"Supported formats are wav, aiff, mp3, flac and ogg.\n\n"

"An exception is thrown when other extensions are given. Note that to encode in mp3 format it is mandatory that ffmpeg was configured with mp3 enabled.");

void AudioWriter::createInnerNetwork() {
  _writer = streaming::AlgorithmFactory::create("AudioWriter");

  // FIXME: 1024 is not the correct size. we should get the exact size from the streaming algorithm, once it has open and close the file. In case it is pcm should be inputBufsize/2/ch or else _audioCtx->frame_size
  _audiogen = new streaming::VectorInput<StereoSample, 1024>();

  _audiogen->output("data")  >>  _writer->input("audio");

  _network = new scheduler::Network(_audiogen);
}

void AudioWriter::configure() {
  try {
    _writer->configure(INHERIT("filename"),
                       INHERIT("format"),
                       INHERIT("sampleRate"));
  }
  catch (EssentiaException&) {
    // no file has been specified, do not do anything
    // we let the inner loader take care of correctness and sending a nice
    // error message if necessary
    _configured = false;
    return;
  }

  _configured = true;

}

void AudioWriter::compute() {
  if (!_configured) {
    throw EssentiaException("AudioWriter: Trying to call compute() on an AudioWriter algo which hasn't been correctly configured...");
  }

  const vector<StereoSample>& audio = _audio.get();

  _audiogen->setVector(&audio);

  _network->run();
}


} // namespace standard
} // namespace essentia
