/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef STREAMINGEXTRACTORTEMPOTAP_H
#define STREAMINGEXTRACTORTEMPOTAP_H

#include "streamingalgorithmcomposite.h"
#include "pool.h"
#include "algorithm.h"
#include "network.h"

namespace essentia {
namespace streaming {

class RhythmExtractor : public AlgorithmComposite {

 protected:
  SinkProxy<Real> _signal;

  Source<Real> _bpm;
  Source<std::vector<Real> > _ticks;
  Source<std::vector<Real> > _estimates;
  Source<std::vector<Real> > _rubatoStart;
  Source<std::vector<Real> > _rubatoStop;
  Source<std::vector<Real> > _bpmIntervals;

  Pool _pool;
  int _frameSize;
  int _hopSize;
  int _zeroPadding;
  Real _sampleRate;
  Real _frameTime;
  Real _tolerance;
  Real _periodTolerance;
  Real _lastBeatInterval;
  int _numberFrames;
  int _frameHop;

  bool _useOnset, _useBands;

  int _preferredBufferSize;

  Algorithm* _frameCutter;
  Algorithm* _windowing;
  Algorithm* _fft;
  Algorithm* _cart2polar;
  Algorithm* _onsetHfc;
  Algorithm* _onsetComplex;
  Algorithm* _spectrum;
  Algorithm* _tempoTapBands;
  Algorithm* _tempoScaleBands;
  Algorithm* _tempoTap;
  Algorithm* _tempoTapTicks;
  standard::Algorithm* _bpmRubato;
  Algorithm* _multiplexer;
  Algorithm* _startStopSilence;
  Algorithm* _derivative;
  Algorithm* _max;
  scheduler::Network* _network;

  bool _configured;
  void createInnerNetwork();
  void clearAlgos();

 public:
  RhythmExtractor();

  ~RhythmExtractor();

  void declareParameters() {
    declareParameter("useOnset", "whether or not to use onsets as periodicity function", "{true,false}", true);
    declareParameter("useBands", "whether or not to use band energy as periodicity function", "{true,false}", true);
    declareParameter("hopSize", "the number of audio samples per features", "(0,inf)", 256);
    declareParameter("frameSize", "the number audio samples used to compute a feature", "(0,inf)", 1024);
    declareParameter("numberFrames", "the number of feature frames to buffer on", "(0,inf)", 1024);
    declareParameter("frameHop", "the number of feature frames separating two evaluations", "(0,inf)", 1024);
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("tolerance", "the minimum interval between two consecutive beats [s]", "[0,inf)", 0.24);
    declareParameter("tempoHints", "the optional list of initial beat locations, to favor the detection of pre-determined tempo period and beats alignment [s]", "", std::vector<Real>());
    declareParameter("maxTempo", "the fastest tempo to detect [bpm]", "[60,250]", 208);
    declareParameter("minTempo", "the slowest tempo to detect [bpm]", "[40,180]", 40);
    declareParameter("lastBeatInterval", "the minimum interval between last beat and end of file [s]", "[0,inf)", 0.100);
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_frameCutter));
    declareProcessStep(SingleShot(this));
  }

  void configure();
  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#include "vectorinput.h"

namespace essentia {
namespace standard {

class RhythmExtractor : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<Real> _bpm;
  Output<std::vector<Real> > _ticks;
  Output<std::vector<Real> > _estimates;
  Output<std::vector<Real> > _rubatoStart;
  Output<std::vector<Real> > _rubatoStop;
  Output<std::vector<Real> > _bpmIntervals;

  bool _configured;

  streaming::Algorithm* _rhythmExtractor;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

 public:

  RhythmExtractor();
  ~RhythmExtractor();

  void declareParameters() {
    declareParameter("useOnset", "whether or not to use onsets as periodicity function", "{true,false}", true);
    declareParameter("useBands", "whether or not to use band energy as periodicity function", "{true,false}", true);
    declareParameter("hopSize", "the number of audio samples per features", "(0,inf)", 256);
    declareParameter("frameSize", "the number audio samples used to compute a feature", "(0,inf)", 1024);
    declareParameter("numberFrames", "the number of feature frames to buffer on", "(0,inf)", 1024);
    declareParameter("frameHop", "the number of feature frames separating two evaluations", "(0,inf)", 1024);
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("tolerance", "the minimum interval between two consecutive beats [s]", "[0,inf)", 0.24);
    declareParameter("tempoHints", "the optional list of initial beat locations, to favor the detection of pre-determined tempo period and beats alignment [s]", "", std::vector<Real>());
    declareParameter("maxTempo", "the fastest tempo to detect [bpm]", "[60,250]", 208);
    declareParameter("minTempo", "the slowest tempo to detect [bpm]", "[40,180]", 40);
    declareParameter("lastBeatInterval", "the minimum interval between last beat and end of file [s]", "[0,inf)", 0.100);
  }

  void configure();
  void compute();
  void createInnerNetwork();
  void reset();

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif // STREAMINGEXTRACTORTEMPOTAP_H
