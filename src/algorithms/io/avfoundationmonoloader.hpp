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
  // Disconnect all null connections to delete the corresponding DevNull objects created.
    // TODO this should be un-commented once https://github.com/MTG/essentia/commit/3a707a08a155bc4899b86cef5bb7c5f3d423834d#diff-ba5f7c2b50a8cc3998f01dd31b0a432967e5771a0fe572ffde82583b18e8523d is included in the release. Until then, disconnecting crashes.
    // Yes, this does mean we have a minor memory leak. Deal with it.
//  disconnect(_audioLoader->output("md5"), NOWHERE);
//  disconnect(_audioLoader->output("bit_rate"), NOWHERE);
//  disconnect(_audioLoader->output("codec"), NOWHERE);
//  disconnect(_audioLoader->output("sampleRate"), NOWHERE);

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
