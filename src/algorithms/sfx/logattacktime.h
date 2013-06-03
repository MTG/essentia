/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_LOGATTACKTIME_H
#define ESSENTIA_LOGATTACKTIME_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class LogAttackTime : public Algorithm {

 private:
  Real _startThreshold, _stopThreshold;

  Input<std::vector<Real> > _signal;
  Output<Real> _logAttackTime;

 public:
  LogAttackTime() {
    declareInput(_signal, "signal", "the input signal envelope (must be non-empty)");
    declareOutput(_logAttackTime, "logAttackTime", "the log (base 10) of the attack time [log10(s)]");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the audio sampling rate [Hz]", "(0,inf)", 44100.);
    declareParameter("startAttackThreshold", "the percentage of the input signal envelope at which the starting point of the attack is considered", "[0,1]", 0.2);
    declareParameter("stopAttackThreshold", "the percentage of the input signal envelope at which the ending point of the attack is considered", "[0,1]", 0.9);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class LogAttackTime : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<Real> _logAttackTime;

  std::vector<Real> _accu;

 public:
  LogAttackTime() {
    declareAlgorithm("LogAttackTime");
    declareInput(_signal, TOKEN, "signal");
    declareOutput(_logAttackTime, TOKEN, "logAttackTime");
  }

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_LOGATTACKTIME_H
