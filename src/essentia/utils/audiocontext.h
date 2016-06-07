/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_AUDIOWRITERTOOL_H
#define ESSENTIA_AUDIOWRITERTOOL_H

#include <string>
#include <vector>
#include "types.h"
#include "ffmpegapi.h"

#define MAX_AUDIO_FRAME_SIZE 192000 // the same value as in AudioLoader

namespace essentia {

/**
 * This is just a nice object-oriented wrapper around FFMPEG
 */
class AudioContext {
 protected:
  bool _isOpen;
  std::string _filename;

  AVStream* _avStream;
  AVFormatContext* _muxCtx;
  AVCodecContext* _codecCtx;

  int _inputBufSize;   // input buffer size
  float* _buffer;      // input FLT buffer interleaved
  uint8_t* _buffer_test; // input buffer in converted to codec sample format

  struct AVAudioResampleContext* _convertCtxAv;

  //const static int FFMPEG_BUFFER_SIZE = MAX_AUDIO_FRAME_SIZE * 2;
  // MAX_AUDIO_FRAME_SIZE is in bytes, multiply it by 2 to get some margin
  

 public:
  AudioContext();
  ~AudioContext() { close(); }
  int create(const std::string& filename, const std::string& format,
             int nChannels, int sampleRate, int bitrate);
  void open();
  bool isOpen() const { return _isOpen; }
  void write(const std::vector<AudioSample>& monoData);
  void write(const std::vector<StereoSample>& stereoData);
  void close();

 protected:
  int16_t scale(Real value);
  void encodePacket(int size);
  void writeEOF();
};

} // namespace essentia

#endif // ESSENTIA_AUDIOWRITERTOOL_H
