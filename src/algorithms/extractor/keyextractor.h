#ifndef KEY_EXTRACTOR_H
#define KEY_EXTRACTOR_H

#include "streamingalgorithmcomposite.h"
#include "pool.h"
#include "algorithm.h"
#include "vectorinput.h"
#include "network.h"

namespace essentia {
namespace streaming {

class KeyExtractor : public AlgorithmComposite {
 protected:
  Algorithm *_frameCutter, *_windowing, *_spectrum, *_spectralPeaks, *_hpcpKey, *_key;
  scheduler::Network* _network;
  bool _configured;

  SinkProxy<Real> _audio;
  SourceProxy<std::string> _keyKey;
  SourceProxy<std::string> _keyScale;
  SourceProxy<Real> _keyStrength;

 public:
  KeyExtractor();
  ~KeyExtractor();

  void declareParameters() {
    declareParameter("frameSize", "the framesize for computing tonal features", "(0,inf)", 4096);
    declareParameter("hopSize", "the hopsize for computing tonal features", "(0,inf)", 2048);
    declareParameter("tuningFrequency", "the tuning frequency of the input signal", "(0,inf)", 440.0);
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_frameCutter));
  }

  void configure();
  void createInnerNetwork();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

namespace essentia {
namespace standard {

class KeyExtractor : public Algorithm {
 protected:
  Input<std::vector<Real> > _audio;
  Output<std::string> _key;
  Output<std::string> _scale;
  Output<Real> _strength;

  bool _configured;

  streaming::Algorithm* _keyExtractor;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

 public:

  KeyExtractor();
  ~KeyExtractor();

  void declareParameters() {
    declareParameter("frameSize", "the framesize for computing tonal features", "(0,inf)", 4096);
    declareParameter("hopSize", "the hopsize for computing tonal features", "(0,inf)", 2048);
    declareParameter("tuningFrequency", "the tuning frequency of the input signal", "(0,inf)", 440.0);
  }

  void configure();
  void createInnerNetwork();
  void compute();
  void reset();

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif // KEY_EXTRACTOR_H
