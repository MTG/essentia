/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef STARTSTOPSILENCE_H
#define STARTSTOPSILENCE_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class StartStopSilence : public Algorithm {

 private:
  Input<std::vector<Real> > _frame;
  Output<int> _startSilenceSource;
  Output<int> _stopSilenceSource;

  int _startSilence;
  int _stopSilence;
  int _nFrame;
  bool _wasSilent;
  Real _threshold;

 public:
  StartStopSilence() {
    declareInput(_frame, "frame", "the input audio frames");
    declareOutput(_startSilenceSource, "startFrame", "number of the first non-silent frame");
    declareOutput(_stopSilenceSource, "stopFrame", "number of the last non-silent frame");
    reset();
  }


  void declareParameters() {
    declareParameter("threshold", "the threshold below which average energy is defined as silence [dB]", "(-inf,0])", -60);
  }

  void configure();
  void compute();
  void reset();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

class StartStopSilence : public Algorithm {

 protected:
  int _startSilence;
  int _stopSilence;
  int _nFrame;
  Real _threshold;

  Source<int> _startSilenceSource;
  Source<int> _stopSilenceSource;
  Sink<std::vector<Real> > _frame;

 public:
  StartStopSilence() {
    declareInput(_frame, 1, "frame", "the input audio frames");
    declareOutput(_startSilenceSource, 0, "startFrame", "number of the first non-silent frame");
    declareOutput(_stopSilenceSource, 0, "stopFrame", "number of the last non-silent frame");
  }

  void declareParameters() {
    declareParameter("threshold", "the threshold below which average energy is defined as silence [dB]", "(-inf,0])", -60);
  }

  void configure();
  AlgorithmStatus process();
  void reset() {
    // TODO: why Algorithm::reset() ?
    Algorithm::reset();
    _startSilence = 0;
    _stopSilence = 0;
    _nFrame = 0;
  }

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#endif // STARTSTOPSILENCE_H
