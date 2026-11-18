#ifndef ESSENTIA_AUDIOCOMMONS_TIMBRAL_MODELS_H
#define ESSENTIA_AUDIOCOMMONS_TIMBRAL_MODELS_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class AudioCommonsTimbralModels : public Algorithm {
 protected:
    Input<std::vector<Real> > _signal;

    Output<Real> _brightness;
    Output<Real> _depth;
    Output<Real> _hardness;
    Output<Real> _roughness;
    Output<Real> _sharpness;
    Output<Real> _warmth;
    Output<Real> _boominess;

    Real _sampleRate;
    bool _computeBrightness;
    bool _computeDepth;
    bool _computeHardness;
    bool _computeRoughness;
    bool _computeSharpness;
    bool _computeWarmth;
    bool _computeBoominess;
    bool _clipOutput;
    bool _loudnessNorm;

 public:
    AudioCommonsTimbralModels()
        : _sampleRate(44100.0f),
          _computeBrightness(true),
          _computeDepth(true),
          _computeHardness(true),
          _computeRoughness(true),
          _computeSharpness(true),
          _computeWarmth(true),
          _computeBoominess(true),
          _clipOutput(false),
          _loudnessNorm(true) {
        declareInput(_signal, "signal", "mono audio input signal buffer (nominally 44.1 kHz)");

        declareOutput(_brightness, "brightness", "apparent brightness score [0-100]");
        declareOutput(_depth, "depth", "apparent depth score [0-100]");
        declareOutput(_hardness, "hardness", "apparent hardness score [0-100]");
        declareOutput(_roughness, "roughness", "apparent roughness score [0-100]");
        declareOutput(_sharpness, "sharpness", "apparent sharpness score [0-100]");
        declareOutput(_warmth, "warmth", "apparent warmth score [0-100]");
        declareOutput(_boominess, "boominess", "apparent boominess score [0-100]");
    }

    void declareParameters() override {
        declareParameter("sampleRate", "the sampling rate of the input audio signal [Hz]", "(0,192000]", 44100.0f);
        declareParameter("computeBrightness", "flag to compute brightness model", "{true,false}", true);
        declareParameter("computeDepth", "flag to compute depth model", "{true,false}", true);
        declareParameter("computeHardness", "flag to compute hardness model", "{true,false}", true);
        declareParameter("computeRoughness", "flag to compute roughness model", "{true,false}", true);
        declareParameter("computeSharpness", "flag to compute sharpness model", "{true,false}", true);
        declareParameter("computeWarmth", "flag to compute warmth model", "{true,false}", true);
        declareParameter("computeBoominess", "flag to compute boominess model", "{true,false}", true);
        declareParameter("clipOutput", "whether to clamp all model scores strictly to [0, 100]", "{true,false}", false);
        declareParameter("loudnessNorm", "whether to normalize audio to -24 LUFS via EBU R128", "{true,false}", true);
    }

    virtual ~AudioCommonsTimbralModels() override = default;

    void configure() override;
    void compute() override;
    void reset() override;

    static const char* name;
    static const char* category;
    static const char* description;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia { 
namespace streaming { 

class AudioCommonsTimbralModels : public StreamingAlgorithmWrapper {
 protected:
  Sink<std::vector<Real> > _signal;
  Source<Real> _brightness;
  Source<Real> _depth;
  Source<Real> _hardness;  
  Source<Real> _roughness;
  Source<Real> _sharpness;  
  Source<Real> _warmth; 
  Source<Real> _boominess;

 public:
  AudioCommonsTimbralModels() {
    declareAlgorithm("AudioCommonsTimbralModels"); 
    declareInput(_signal, TOKEN, "signal"); 
    declareOutput(_brightness, TOKEN, "brightness");
    declareOutput(_depth, TOKEN, "depth");
    declareOutput(_hardness, TOKEN, "hardness");  
    declareOutput(_roughness, TOKEN, "roughness");
    declareOutput(_sharpness, TOKEN, "sharpness");  
    declareOutput(_warmth, TOKEN, "warmth");  
    declareOutput(_boominess, TOKEN, "boominess");
  } 
}; 

} // namespace streaming                                                                                                                                                               
} // namespace essentia                                                                                                                                                                

#endif // ESSENTIA_AUDIOCOMMONS_TIMBRAL_MODELS_H
