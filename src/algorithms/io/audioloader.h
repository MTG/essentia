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

#ifndef ESSENTIA_STREAMING_AUDIOLOADER_H
#define ESSENTIA_STREAMING_AUDIOLOADER_H

#include "streamingalgorithm.h"
#include "network.h"
#include "ffmpegapi.h"

#define MAX_AUDIO_FRAME_SIZE 192000

namespace essentia {
namespace streaming {

class AudioLoader : public Algorithm {
 protected:
  Source<StereoSample> _audio;
  AbsoluteSource<Real> _sampleRate;
  AbsoluteSource<int> _channels;
  int _nChannels;

  // MAX_AUDIO_FRAME_SIZE is in bytes, we want FFMPEG_BUFFER_SIZE in sample units
  // we also multiply by 2 to get some margin, because we might want to decode multiple frames
  // in this buffer (all the frames contained in a packet, which can be more than 1 as in flac),
  // and each time we decode a frame we need to have at least a full buffer of free space.
  const static int FFMPEG_BUFFER_SIZE = (MAX_AUDIO_FRAME_SIZE / sizeof(int16_t)) * 2;

  int16_t* _buffer;
  int _dataSize;

  AVFormatContext* _demuxCtx;
  AVCodecContext* _audioCtx;
  AVCodec* _audioCodec;
  AVPacket _packet;

#if LIBAVCODEC_VERSION_INT >= AVCODEC_AUDIO_DECODE4
  AVFrame* _decodedFrame;
#endif

#if HAVE_SWRESAMPLE
  struct SwrContext* _convertCtx;
#else
  AVAudioConvert* _audioConvert;
  int16_t* _buff1;
  int16_t* _buff2;
#endif

  int _streamIdx; // index of the audio stream among all the streams contained in the file
  bool _configured;


  void openAudioFile(const std::string& filename);
  void closeAudioFile();

  void pushChannelsSampleRateInfo(int nChannels, Real sampleRate);
  int decode_audio_frame(AVCodecContext* audioCtx, int16_t* output,
                         int* outputSize, AVPacket* packet);
  int decodePacket();
  void flushPacket();
  void copyFFmpegOutput();


 public:
  AudioLoader() : Algorithm(), _buffer(0),  _demuxCtx(0),
	          _audioCtx(0), _audioCodec(0),
#if LIBAVCODEC_VERSION_INT >= AVCODEC_AUDIO_DECODE4
                  _decodedFrame(0),
#endif
#if HAVE_SWRESAMPLE
                  _convertCtx(0),
#else
                  _audioConvert(0), _buff1(0), _buff2(0),
#endif
                  _configured(false) {

    declareOutput(_audio, 1, "audio", "the input audio signal");
    declareOutput(_sampleRate, 0, "sampleRate", "the sampling rate of the audio signal [Hz]");
    declareOutput(_channels, 0, "numberChannels", "the number of channels");

    _audio.setBufferType(BufferUsage::forLargeAudioStream);

    // Register all formats and codecs
    av_register_all();

    // use av_malloc, because we _need_ the buffer to be 16-byte aligned
    _buffer = (int16_t*)av_malloc(FFMPEG_BUFFER_SIZE * sizeof(int16_t));
  }

  ~AudioLoader();

  AlgorithmStatus process();
  void reset();

  void declareParameters() {
    declareParameter("filename", "the name of the file from which to read", "", Parameter::STRING);
  }

  void configure();

  static const char* name;
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
class AudioLoader : public Algorithm {

 protected:
  Output<std::vector<StereoSample> > _audio;
  Output<Real> _sampleRate;
  Output<int> _channels;

  streaming::Algorithm* _loader;
  streaming::VectorOutput<StereoSample>* _audioStorage;
  streaming::VectorOutput<Real>* _srStorage;
  streaming::VectorOutput<int>* _cStorage;
  std::vector<Real> _sampleRateStorage;
  std::vector<int> _channelsStorage;

  scheduler::Network* _network;

  void createInnerNetwork();

 public:
  AudioLoader() {
    declareOutput(_audio, "audio", "the input audio signal");
    declareOutput(_sampleRate, "sampleRate", "the sampling rate of the audio signal [Hz]");
    declareOutput(_channels, "numberChannels", "the number of channels");

    createInnerNetwork();
  }

  ~AudioLoader() {
    // NB: this will also delete all the algorithms as the Network took ownership of them
    delete _network;
  }

  void declareParameters() {
    declareParameter("filename", "the name of the file from which to read", "", Parameter::STRING);
  }

  void configure();

  void compute();
  void reset();

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif // ESSENTIA_STREAMING_AUDIOLOADER_H
