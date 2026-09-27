#ifndef AVFoundationAVMonoLoader_hpp
#define AVFoundationAVMonoLoader_hpp

#include "streamingalgorithmcomposite.h"
#include "network.h"

namespace essentia {
namespace streaming {

class AVMonoLoader : public AlgorithmComposite {
 protected:
  Algorithm* _audioLoader;
  Algorithm* _mixer;
  Algorithm* _resample;

  SourceProxy<AudioSample> _audio;
  bool _configured;

 public:
  AVMonoLoader();

  ~AVMonoLoader() {
  disconnect(_audioLoader->output("md5"), NOWHERE);
  disconnect(_audioLoader->output("bit_rate"), NOWHERE);
  disconnect(_audioLoader->output("codec"), NOWHERE);
  disconnect(_audioLoader->output("sampleRate"), NOWHERE);

  delete _audioLoader;
  delete _mixer;
  delete _resample;
  }

  void declareParameters() {
  declareParameter("filename", "the name of the file from which to read", "", Parameter::STRING);
  declareParameter("sampleRate", "the desired output sampling rate [Hz]", "(0,inf)", 44100.);
  declareParameter("downmix", "the mixing type for stereo files", "{left,right,mix}", "mix");
  declareParameter("audioStream", "audio stream index to be loaded. Other streams are no taken into account (e.g. if stream 0 is video and 1 is audio use index 0 to access it.)", "[0,inf)", 0);

  }

  void declareProcessOrder() {
  declareProcessStep(ChainFrom(_audioLoader));
  }

  void configure();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#endif /* AVFoundationAVMonoLoader_hpp */
