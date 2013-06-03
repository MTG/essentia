/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_DURATION_H
#define ESSENTIA_DURATION_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Duration : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<Real> _duration;

 public:
  Duration() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_duration, "duration", "the duration of the signal [s]");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
  }

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia


#include "accumulatoralgorithm.h"

namespace essentia {
namespace streaming {

class Duration : public AccumulatorAlgorithm {

 protected:
  Sink<Real> _signal;
  Source<Real> _duration;

  uint64 _nsamples;

 public:
  Duration() : _nsamples(0) {
    declareInputStream(_signal, "signal", "the input signal");
    declareOutputResult(_duration, "duration", "the duration of the signal [s]");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
  }

  void reset();

  void consume();
  void finalProduce();

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_DURATION_H
