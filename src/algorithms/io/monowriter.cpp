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

#include "monowriter.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* MonoWriter::name = "MonoWriter";
const char* MonoWriter::description = DOC("This algorithm writes a mono audio stream to a file.\n\n"

"Supported formats are wav, aiff, mp3, flac and ogg. An exception is thrown when other extensions are given. Note that to encode in mp3 format it is mandatory that ffmpeg was configured with mp3 enabled.\n\n"

"If the file specified by filename could not be opened or the header of the file omits channel's information, an exception is thrown.");


void MonoWriter::reset() {
  Algorithm::reset();
  int recommendedBufferSize;
  try {
    recommendedBufferSize = _audioCtx.create(parameter("filename").toString(),
                                             parameter("format").toString(),
                                             1, // nChannels
                                             parameter("sampleRate").toInt(),
                                             parameter("bitrate").toInt()*1000);
  }
  catch (EssentiaException& e) {
    throw EssentiaException("MonoWriter: Error creating audio file: ", e.what());
  }

  _audio.setAcquireSize(recommendedBufferSize);
  _audio.setReleaseSize(recommendedBufferSize);
}


void MonoWriter::configure() {
  if (!parameter("filename").isConfigured() || parameter("filename").toString().empty()) {
    // no file has been specified or retarded name specified, do nothing
    _configured = false;
    return;
  }

  reset();
  _configured = true;
}


AlgorithmStatus MonoWriter::process() {
  if (!_configured) {
    throw EssentiaException("MonoWriter: Trying to call process() on an MonoWriter algo which hasn't been correctly configured");
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
    throw EssentiaException("MonoWriter: error writing to audio file: ", e.what());
  }

  releaseData();

  return OK;
}


} // namespace streaming
} // namespace essentia


#include "algorithmfactory.h"

namespace essentia {
namespace standard {

const char* MonoWriter::name = "MonoWriter";
const char* MonoWriter::description = DOC("This algorithm writes a mono audio stream to a file.\n\n"

"Supported formats are wav, aiff, mp3, flac and ogg. An exception is thrown when other extensions are given. Note that to encode in mp3 format it is mandatory that ffmpeg was configured with mp3 enabled.\n\n"

"If the file specified by filename could not be opened or the header of the file omits channel's information, an exception is thrown.");


void MonoWriter::createInnerNetwork() {
  _writer = streaming::AlgorithmFactory::create("MonoWriter");
  _audiogen = new streaming::VectorInput<AudioSample, 1024>();

  _audiogen->output("data")  >>  _writer->input("audio");

  _network = new scheduler::Network(_audiogen);
}

void MonoWriter::configure() {
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

void MonoWriter::compute() {
  if (!_configured) {
    throw EssentiaException("MonoWriter: Trying to call compute() on an MonoWriter algo which hasn't been correctly configured...");
  }

  const vector<AudioSample>& audio = _audio.get();

  _audiogen->setVector(&audio);
  _network->run();

  // TODO: should we reset it here, same as MonoLoader to allow it to write twice in a row?
}


} // namespace standard
} // namespace essentia
