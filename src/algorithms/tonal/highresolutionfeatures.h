/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_HIGHRESOLUTIONFEATURES_H
#define ESSENTIA_HIGHRESOLUTIONFEATURES_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class HighResolutionFeatures : public Algorithm {

 protected:
  Input<std::vector<Real> > _hpcp;
  Output<Real> _equalTemperedDeviation;
  Output<Real> _nt2tEnergyRatio;
  Output<Real> _nt2tPeaksEnergyRatio;

 public:

  HighResolutionFeatures() {
    declareInput(_hpcp, "hpcp", "the HPCPs, preferably of size >= 120");

    declareOutput(_equalTemperedDeviation, "equalTemperedDeviation",
                   "measure of the deviation of HPCP local maxima with respect to equal-tempered bins");
    declareOutput(_nt2tEnergyRatio, "nonTemperedEnergyRatio",
                   "ratio between the energy on non-tempered bins and the total energy");
    declareOutput(_nt2tPeaksEnergyRatio, "nonTemperedPeaksEnergyRatio",
                   "ratio between the energy on non-tempered peaks and the total energy");
  }

  void declareParameters() {
    declareParameter("maxPeaks", "maximum number of HPCP peaks to consider when calculating outputs", "[1,inf)", 24);
  }

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmcomposite.h"
#include "pool.h"

namespace essentia {
namespace streaming {

class HighResolutionFeatures : public AlgorithmComposite {

 protected:
  SinkProxy<std::vector<Real> > _pcp;
  Source<Real> _equalTemperedDeviation;
  Source<Real> _nt2tEnergyRatio;
  Source<Real> _nt2tPeaksEnergyRatio;

  Pool _pool;
  Algorithm* _poolStorage;
  standard::Algorithm* _highResAlgo;

 public:
  HighResolutionFeatures();
  ~HighResolutionFeatures() {
    delete _highResAlgo;
    delete _poolStorage;
  }

  void declareParameters() {
    declareParameter("maxPeaks", "maximum number of HPCP peaks to consider when calculating outputs", "[1,inf)", 24);
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_poolStorage));
    declareProcessStep(SingleShot(this));
  }

  void configure();
  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* description;

};


} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_HIGHRESOLUTIONFEATURES_H
