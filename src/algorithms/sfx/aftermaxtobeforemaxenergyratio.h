/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_AFTERMAXTOBEFOREMAXENERGYRATIO_H
#define ESSENTIA_AFTERMAXTOBEFOREMAXENERGYRATIO_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class AfterMaxToBeforeMaxEnergyRatio : public Algorithm {

 protected:
  Input<std::vector<Real> > _pitch;
  Output<Real> _afterMaxToBeforeMaxEnergyRatio;

 public:
  AfterMaxToBeforeMaxEnergyRatio() {
    declareInput(_pitch, "pitch", "the array of pitch values [Hz]");
    declareOutput(_afterMaxToBeforeMaxEnergyRatio, "afterMaxToBeforeMaxEnergyRatio",
                  "the ratio between the pitch energy after the pitch maximum to the pitch energy before the pitch maximum");
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

namespace essentia {
namespace streaming {

class AfterMaxToBeforeMaxEnergyRatio : public Algorithm {

 protected:
  Sink<Real> _pitch;
  Source<Real> _afterMaxToBeforeMaxEnergyRatio;

  std::vector<Real> _accu;

 public:
  AfterMaxToBeforeMaxEnergyRatio() {
    declareInput(_pitch, 1, "pitch", "the array of pitch values [Hz]");
    declareOutput(_afterMaxToBeforeMaxEnergyRatio, 0, "afterMaxToBeforeMaxEnergyRatio",
                  "the ratio between the pitch energy after the pitch maximum to the pitch energy \
                  before the pitch maximum");
  }

  AlgorithmStatus process();
  void declareParameters() {}
  void reset() {
    Algorithm::reset();
    _accu.clear();
  }

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_AFTERMAXTOBEFOREMAXENERGYRATIO_H
