/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_AUDIOWRITERTOOL_H
#define ESSENTIA_AUDIOWRITERTOOL_H

#include <string>
#include <vector>
#include "types.h"
#include "ffmpegapi.h"


namespace essentia {

/**
 * This is just a nice object-oriented wrapper around FFMPEG
 */
class AudioContext {
 protected:
  bool _isOpen;
  std::string _filename;

  AVStream* _avStream;
  AVFormatContext* _demuxCtx;
  AVCodecContext* _codecCtx;

  int _outputBufSize; // (frame)size of output buffer
  int _inputBufSize;     // input buffer size
  int16_t* _inputBuffer; // input buffer interleaved
  uint8_t* _outputBuffer; // output buffer interleaved

  bool _isFlac;

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
  static const int SAMPLE_SIZE_RATIO;
};

} // namespace essentia

#endif // ESSENTIA_AUDIOWRITERTOOL_H
