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

#ifndef ESSENTIA_STREAMING_AUDIOLOADER_H
#define ESSENTIA_STREAMING_AUDIOLOADER_H

#include "streamingalgorithm.h"
#include "network.h"
#include "ffmpegapi.h"
#include "poolstorage.h"


#define MAX_AUDIO_FRAME_SIZE 192000

namespace essentia {
namespace streaming {

class AudioLoader : public Algorithm {
 protected:
  Source<StereoSample> _audio;
  AbsoluteSource<Real> _sampleRate;
  AbsoluteSource<int> _channels;
  AbsoluteSource<std::string> _md5;
  AbsoluteSource<int> _bit_rate;
  AbsoluteSource<std::string> _codec;

  int _nChannels;

  // MAX_AUDIO_FRAME_SIZE is in bytes, multiply it by 2 to get some margin, 
  // because we might want to decode multiple frames in this buffer (all the 
  // frames contained in a packet, which can be more than 1 as in flac), and 
  // each time we decode a frame we need to have at least a full buffer of free space.
  const static int FFMPEG_BUFFER_SIZE = MAX_AUDIO_FRAME_SIZE * 2;

  float* _buffer;
  int _dataSize;

  AVFormatContext* _demuxCtx;
  AVCodecContext* _audioCtx;
  AVCodec* _audioCodec;
  AVPacket _packet;
  AVMD5 *_md5Encoded;
  uint8_t _checksum[16];
  bool _computeMD5;
  AVFrame* _decodedFrame;

  struct SwrContext* _convertCtxAv;

  int _streamIdx; // index of the audio stream among all the streams contained in the file
  std::vector<int> _streams;
  int _selectedStream;
  bool _configured;


  void openAudioFile(const std::string& filename);
  void closeAudioFile();

  void pushChannelsSampleRateInfo(int nChannels, Real sampleRate);
  void pushCodecInfo(std::string codec, int bit_rate);
  int decode_audio_frame(AVCodecContext* audioCtx, float* output,
                         int* outputSize, AVPacket* packet);
  int decodePacket();
  void flushPacket();
  void copyFFmpegOutput();


 public:
  AudioLoader() : Algorithm(), _buffer(0),  _demuxCtx(0),
	          _audioCtx(0), _audioCodec(0), _decodedFrame(0),
            _convertCtxAv(0), _configured(false) {

    declareOutput(_audio, 1, "audio", "the input audio signal");
    declareOutput(_sampleRate, 0, "sampleRate", "the sampling rate of the audio signal [Hz]");
    declareOutput(_channels, 0, "numberChannels", "the number of channels");
    declareOutput(_md5, 0, "md5", "the MD5 checksum of raw undecoded audio payload");
    declareOutput(_bit_rate, 0, "bit_rate", "the bit rate of the input audio, as reported by the decoder codec");
    declareOutput(_codec, 0, "codec", "the codec that is used to decode the input audio");

    _audio.setBufferType(BufferUsage::forLargeAudioStream);

    // Register all formats and codecs
    av_register_all();

    // use av_malloc, because we _need_ the buffer to be 16-byte aligned
    _buffer = (float*)av_malloc(FFMPEG_BUFFER_SIZE);

    _md5Encoded = av_md5_alloc();
    if (!_md5Encoded) {
        throw EssentiaException("Error allocating the MD5 context");
    }
  }

  ~AudioLoader();

  AlgorithmStatus process();
  void reset();

  void declareParameters() {
    declareParameter("filename", "the name of the file from which to read", "", Parameter::STRING);
    declareParameter("computeMD5", "compute the MD5 checksum", "{true,false}", false);
    declareParameter("audioStream", "audio stream index to be loaded. Other streams are not taken into account (e.g. if stream 0 is video and 1 is audio use index 0 to access it.)", "[0,inf)", 0);
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
class AudioLoader : public Algorithm {

 protected:
  Output<std::vector<StereoSample> > _audio;
  Output<Real> _sampleRate;
  Output<int> _channels;
  Output<std::string> _md5;
  Output<int> _bit_rate;
  Output<std::string> _codec;

  streaming::Algorithm* _loader;
  streaming::VectorOutput<StereoSample>* _audioStorage;

  scheduler::Network* _network;
  Pool _pool;

  void createInnerNetwork();

 public:
  AudioLoader() {
    declareOutput(_audio, "audio", "the input audio signal");
    declareOutput(_sampleRate, "sampleRate", "the sampling rate of the audio signal [Hz]");
    declareOutput(_channels, "numberChannels", "the number of channels");
    declareOutput(_md5, "md5", "the MD5 checksum of raw undecoded audio payload");
    declareOutput(_bit_rate, "bit_rate", "the bit rate of the input audio, as reported by the decoder codec");
    declareOutput(_codec, "codec", "the codec that is used to decode the input audio");

    createInnerNetwork();
  }

  ~AudioLoader() {
    // NB: this will also delete all the algorithms as the Network took ownership of them
    delete _network;
  }

  void declareParameters() {
    declareParameter("filename", "the name of the file from which to read", "", Parameter::STRING);
    declareParameter("computeMD5", "compute the MD5 checksum", "{true,false}", false);
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

#endif // ESSENTIA_STREAMING_AUDIOLOADER_H
