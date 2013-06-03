/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_TEMPOTAPTICKS_H
#define ESSENTIA_TEMPOTAPTICKS_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class TempoTapTicks : public Algorithm {

 private:
  Input< std::vector<Real> > _periods;
  Input< std::vector<Real> > _phases;
  Output<std::vector<Real> > _ticks;
  Output<std::vector<Real> > _matchingPeriods;
  Real _frameTime;
  Real _sampleRate;
  int _nextPhase;
  int _frameHop;
  int _nframes;
  Real _periodTolerance;
  Real _phaseTolerance;

 public:
  TempoTapTicks() {
    declareInput(_periods, "periods", "tempo period candidates for the current frame, in frames");
    declareInput(_phases, "phases", "tempo ticks phase candidates for the current frame, in frames");
    declareOutput(_ticks, "ticks", "the list of resulting ticks [s]");
    declareOutput(_matchingPeriods, "matchingPeriods", "list of matching periods [s]");
  }

  ~TempoTapTicks() {};

  void declareParameters() {
    declareParameter("frameHop", "number of feature frames separating two evaluations", "(0,inf)", 512);
    declareParameter("hopSize", "number of audio samples per features", "(0,inf)", 256);
    declareParameter("sampleRate", "sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
  }

  void configure();
  void compute();
  void reset();

  static const char* name;
  static const char* description;

}; // class TempoTapTicks

} // namespace essentia
} // namespace standard

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class TempoTapTicks : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _periods;
  Sink<std::vector<Real> > _phases;
  Source<std::vector<Real> > _ticks;
  Source<std::vector<Real> > _matchingPeriods;

 public:
  TempoTapTicks() {
    declareAlgorithm("TempoTapTicks");
    declareInput(_periods, TOKEN, "periods");
    declareInput(_phases, TOKEN, "phases");
    declareOutput(_ticks, TOKEN, "ticks");
    declareOutput(_matchingPeriods, TOKEN, "matchingPeriods");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_TEMPOTAPTICKS_H
