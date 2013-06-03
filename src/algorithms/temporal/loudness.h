/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_LOUDNESS_H
#define ESSENTIA_LOUDNESS_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Loudness : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<Real> _loudness;

 public:
  Loudness() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_loudness, "loudness", "the loudness of the input signal");
  }

  void declareParameters() {}
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Loudness : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<Real> _loudness;

 public:
  Loudness() {
    declareAlgorithm("Loudness");
    declareInput(_signal, TOKEN, "signal");
    declareOutput(_loudness, TOKEN, "loudness");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_LOUDNESS_H
