/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_FLATNESSSFX_H
#define ESSENTIA_FLATNESSSFX_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class FlatnessSFX : public Algorithm {

 protected:
  Input<std::vector<Real> > _envelope;
  Output<Real> _flatnessSFX;

 public:
  FlatnessSFX() {
    declareInput(_envelope, "envelope", "the envelope of the signal");
    declareOutput(_flatnessSFX, "flatness", "the flatness coefficient");
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;

  // these thresholds are given in percentage of the total signal length
  // they are used to determine the values that are at the lower threshold (5%)
  // and the upper threshold (80%) respectively
  static const Real lowerThreshold;
  static const Real upperThreshold;

 private:
  Real rollOff(const std::vector<Real>& envelope, Real x) const;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class FlatnessSFX : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _envelope;
  Source<Real> _flatnessSFX;

 public:
  FlatnessSFX() {
    declareAlgorithm("FlatnessSFX");
    declareInput(_envelope, TOKEN, "envelope");
    declareOutput(_flatnessSFX, TOKEN, "flatness");
  }

};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_FLATNESSSFX_H
