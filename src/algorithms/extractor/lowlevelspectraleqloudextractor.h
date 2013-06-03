#ifndef LOW_LEVEL_SPECTRAL_EQLOUD_EXTRACTOR_H
#define LOW_LEVEL_SPECTRAL_EQLOUD_EXTRACTOR_H

#include "streamingalgorithmcomposite.h"
#include "pool.h"
#include "algorithm.h"
#include "vectorinput.h"
#include "network.h"

namespace essentia {
namespace streaming {

class LowLevelSpectralEqloudExtractor : public AlgorithmComposite {
 protected:
  SinkProxy<Real> _signal;

  SourceProxy<Real> _scentroid;
  SourceProxy<Real> _dissonanceValue;
  SourceProxy<std::vector<Real> > _sccontrast;
  SourceProxy<std::vector<Real> > _scvalleys;
  SourceProxy<Real> _kurtosis;
  SourceProxy<Real> _skewness;
  SourceProxy<Real> _spread;

  Algorithm *_centralMoments, *_centroid, *_dissonance, *_distributionShape,
             *_frameCutter, *_spectralContrast, *_spectralPeaks, *_spectrum,
             *_square, *_windowing;

  scheduler::Network* _network;

  bool _configured;
  void clearAlgos();

 public:
  LowLevelSpectralEqloudExtractor();
  ~LowLevelSpectralEqloudExtractor();

  void declareParameters() {
    declareParameter("frameSize", "the frame size for computing low level features", "(0,inf)", 2048);
    declareParameter("hopSize", "the hop size for computing low level features", "(0,inf)", 1024);
    declareParameter("sampleRate", "the audio sampling rate", "(0,inf)", 44100.0);
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

class LowLevelSpectralEqloudExtractor : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _dissonance;
  Output<std::vector<std::vector<Real> > > _sccoeffs;
  Output<std::vector<std::vector<Real> > > _scvalleys;
  Output<std::vector<Real> > _centroid;
  Output<std::vector<Real> > _kurtosis;
  Output<std::vector<Real> > _skewness;
  Output<std::vector<Real> > _spread;

  bool _configured;

  streaming::Algorithm* _lowlevelExtractor;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

 public:

  LowLevelSpectralEqloudExtractor();
  ~LowLevelSpectralEqloudExtractor();

  void declareParameters() {
    declareParameter("frameSize", "the frame size for computing low level features", "(0,inf)", 2048);
    declareParameter("hopSize", "the hop size for computing low level features", "(0,inf)", 1024);
    declareParameter("sampleRate", "the audio sampling rate", "(0,inf)", 44100.0);
  }

  void configure();
  void createInnerNetwork();
  void compute();
  void reset();

  static const char* name;
  static const char* description;
};

}
}





#endif
