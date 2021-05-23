#ifndef AVFoundationLoader_hpp
#define AVFoundationLoader_hpp

#include "streamingalgorithm.h"
#include "network.h"
#include "poolstorage.h"
#include "AVFoundationAudioFile.hpp"

#define MAX_AUDIO_FRAME_SIZE 192000

namespace essentia {
namespace streaming {

class AVAudioLoader : public Algorithm {
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
  const static size_t BUFFER_SIZE = MAX_AUDIO_FRAME_SIZE * 2;

  AVFoundationAudioFile *_file;
  uint8_t _checksum[16];
  bool _computeMD5;
  
  struct AVAudioResampleContext* _convertCtxAv;

  int _streamIdx; // index of the audio stream among all the streams contained in the file
  std::vector<int> _streams;
  int _selectedStream;
  bool _configured;


  void openAudioFile(const std::string& filename);
  void closeAudioFile();

  void pushChannelsSampleRateInfo(int nChannels, Real sampleRate);
  void pushCodecInfo(std::string codec, int bit_rate);

  void copyOutput();

 public:
  AVAudioLoader() : Algorithm(), _configured(false) {

	declareOutput(_audio, 1, "audio", "the input audio signal");
	declareOutput(_sampleRate, 0, "sampleRate", "the sampling rate of the audio signal [Hz]");
	declareOutput(_channels, 0, "numberChannels", "the number of channels");
	declareOutput(_md5, 0, "md5", "the MD5 checksum of raw undecoded audio payload");
	declareOutput(_bit_rate, 0, "bit_rate", "the bit rate of the input audio, as reported by the decoder codec");
	declareOutput(_codec, 0, "codec", "the codec that is used to decode the input audio");

	_audio.setBufferType(BufferUsage::forLargeAudioStream);
  }

  ~AVAudioLoader();

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


#endif /* AVFoundationLoader_hpp */
