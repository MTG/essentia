/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_RMS_H
#define ESSENTIA_RMS_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class RMS : public Algorithm {

 private:
  Input<std::vector<Real> > _array;
  Output<Real> _rms;

 public:
  RMS() {
    declareInput(_array, "array", "the input array");
    declareOutput(_rms, "rms", "the root mean square of the input array");
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

class RMS : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<Real> _rms;

 public:
  RMS() {
    declareAlgorithm("RMS");
    declareInput(_array, TOKEN, "array");
    declareOutput(_rms, TOKEN, "rms");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_RMS_H
