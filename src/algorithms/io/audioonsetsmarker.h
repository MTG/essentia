/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_AUDIOONSETSMARKER_H
#define ESSENTIA_AUDIOONSETSMARKER_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class AudioOnsetsMarker : public Algorithm {

 protected:
  Input<std::vector<Real> > _input;
  Output<std::vector<Real> > _output;

  Real _sampleRate;
  std::vector<Real> _onsets;

  bool _beep;

 public:
  AudioOnsetsMarker() : _beep(false) {
    declareInput(_input, "signal", "the input signal");
    declareOutput(_output, "signal", "the input signal mixed with bursts at onset locations");
  }

  ~AudioOnsetsMarker() {}

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the output signal [Hz]", "(0,inf)", 44100.);
    declareParameter("type", "the type of sound to be added on the event", "{beep,noise}", "beep");
    declareParameter("onsets", "the list of onset locations [s]", "", std::vector<Real>());
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia


#include "streamingalgorithm.h"
#include "network.h"

namespace essentia {
namespace streaming {

class AudioOnsetsMarker : public Algorithm {
 protected:
  Sink<Real> _input;
  Source<Real> _output;

  Real _sampleRate;
  std::vector<Real> _burst;
  std::vector<Real> _onsets;
  bool _beep;
  int _onsetIdx, _burstIdx;
  int _processedSamples;
  int _preferredSize;

 public:
  AudioOnsetsMarker();
  ~AudioOnsetsMarker() {}

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the output signal [Hz]", "(0,inf)", 44100.);
    declareParameter("type", "the type of sound to be added on the event", "{beep,noise}", "beep");
    declareParameter("onsets", "the list of onset locations [s]", "", std::vector<Real>());
  }

  void createInnerNetwork();
  void configure();
  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_AUDIOONSETSMARKER_H
