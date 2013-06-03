/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_LPC_H
#define ESSENTIA_LPC_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class LPC : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _lpc;
  Output<std::vector<Real> > _reflection;
  Algorithm* _correlation;
  std::vector<Real> _r;
  int _P;

 public:
  LPC() : _correlation(0) {
    declareInput(_signal, "frame", "the input audio frame");
    declareOutput(_lpc, "lpc", "the LPC coefficients");
    declareOutput(_reflection, "reflection", "the reflection coefficients");
  }

  ~LPC() {
    delete _correlation;
  }

  void declareParameters() {
    declareParameter("order", "the order of the LPC analysis (typically [8,14])", "[2,inf)", 10);
    declareParameter("type", "the type of LPC (regular or warped)", "{regular,warped}", "regular");
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
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

class LPC : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<std::vector<Real> > _lpc;
  Source<std::vector<Real> > _reflection;

 public:
  LPC() {
    declareAlgorithm("LPC");
    declareInput(_signal, TOKEN, "frame");
    declareOutput(_lpc, TOKEN, "lpc");
    declareOutput(_reflection, TOKEN, "reflection");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_LPC_H
