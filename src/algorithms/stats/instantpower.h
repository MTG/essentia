/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_INSTANTPOWER_H
#define ESSENTIA_INSTANTPOWER_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class InstantPower : public Algorithm {

 private:
  Input<std::vector<Real> > _array;
  Output<Real> _power;

 public:
  InstantPower() {
    declareInput(_array, "array", "the input array");
    declareOutput(_power, "power", "the instant power of the input array");
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

class InstantPower : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<Real> _rms;

 public:
  InstantPower() {
    declareAlgorithm("InstantPower");
    declareInput(_array, TOKEN, "array");
    declareOutput(_rms, TOKEN, "power");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_INSTANTPOWER_H
