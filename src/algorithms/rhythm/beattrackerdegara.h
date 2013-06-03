/*
 * Copyright (C) 2006-2013 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef BEATTRACKERDEGARA_H
#define BEATTRACKERDEGARA_H

#include "streamingalgorithmcomposite.h"
#include "pool.h"
#include "algorithm.h"
#include "network.h"

namespace essentia {
namespace streaming {

class BeatTrackerDegara : public AlgorithmComposite {

 protected:
  SinkProxy<Real>_signal;
  SourceProxy<Real> _ticks;

  Pool _pool;

  // algorithm numeration corresponds to the process chains 
  Algorithm* _frameCutter;
  Algorithm* _windowing;
  Algorithm* _fft;
  Algorithm* _cart2polar;
  Algorithm* _onsetComplex;
  Algorithm* _ticksComplex;
  
  scheduler::Network* _network;
  bool _configured;

  void createInnerNetwork();
  void clearAlgos();
  Real _sampleRate;

 public:
  BeatTrackerDegara();
  ~BeatTrackerDegara();

  void declareParameters() {
    //declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("maxTempo", "the fastest tempo to detect [bpm]", "[60,250]", 208);
    declareParameter("minTempo", "the slowest tempo to detect [bpm]", "[40,180]", 40);
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_frameCutter));  
  }

  void configure();
  void reset();

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#include "vectorinput.h"

namespace essentia {
namespace standard {

class BeatTrackerDegara : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _ticks;

  streaming::Algorithm* _beatTracker;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

 public:

  BeatTrackerDegara();
  ~BeatTrackerDegara();

  void declareParameters() {
    //declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("maxTempo", "the fastest tempo to detect [bpm]", "[60,250]", 208);
    declareParameter("minTempo", "the slowest tempo to detect [bpm]", "[40,180]", 40);
  }

  void configure();
  void compute();
  void reset();
  void createInnerNetwork();

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif // BEATTRACKERDEGARA_H
