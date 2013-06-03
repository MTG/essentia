#include "audioengine.h"
#include "inputcopy.h"
#include "RtAudio.h"

using essentia::streaming::InputCopy;

const unsigned int nChannels = 2;

class AudioEngineImpl
{
  public:
    InputCopy* inputCopy;
    RtAudio rtAudio;

    AudioEngineImpl(InputCopy* inputCopy)
    :inputCopy(inputCopy)
    {
      int iDevice = 0;
      int oDevice = 0;

      RtAudio::DeviceInfo info;

      unsigned int devices = rtAudio.getDeviceCount();
      for (unsigned int i=0; i<devices; i++) {
        info = rtAudio.getDeviceInfo(i);
        
        if (iDevice==0 && info.isDefaultInput) iDevice = i;
        if (oDevice==0 && info.isDefaultOutput) oDevice = i;
        
        if (info.name.find("Soundflower")!=std::string::npos) {
          if (info.inputChannels==nChannels) {
            iDevice = i;
          }
        }
        if (info.name.find("Built-in")!=std::string::npos) {
          if (info.outputChannels==nChannels) {
            oDevice = i;
          }
        }
      }
      
      unsigned int bufferFrames = 1024;
      RtAudio::StreamParameters iParams, oParams;
      iParams.deviceId = iDevice;
      iParams.nChannels = nChannels;
      iParams.firstChannel = 0;
      oParams.deviceId = oDevice;
      oParams.nChannels = nChannels;
      oParams.firstChannel = 0;

      RtAudio::StreamOptions options;
      
      float sampleRate = 44100; 

      rtAudio.openStream( &oParams, &iParams, RTAUDIO_FLOAT32, sampleRate, &bufferFrames, &_callback, (void *)this, &options );
    }
    ~AudioEngineImpl()
    {
      rtAudio.stopStream();
      rtAudio.closeStream();
    }
    void run()
    {
      rtAudio.startStream();
    }
    static int _callback(void *outputBuffer, void *inputBuffer, 
      unsigned int nBufferFrames,
      double streamTime, RtAudioStreamStatus status, void *data )
    {
      ((AudioEngineImpl*)data)->callback((float*)outputBuffer,(float*)inputBuffer,nBufferFrames);
      return 0;
    }
    void callback(float* output,float* input,unsigned int nFrames)
    {
      inputCopy->setInput(input,nFrames*nChannels);
      runTask(inputCopy);
      memcpy(output,input,nChannels*nFrames*sizeof(float));
    }
};

AudioEngine::AudioEngine(essentia::streaming::InputCopy* inputCopy)
{
  impl = new AudioEngineImpl(inputCopy);
}

AudioEngine::~AudioEngine()
{
  delete impl;
}

void AudioEngine::run()
{
  impl->run();
}

