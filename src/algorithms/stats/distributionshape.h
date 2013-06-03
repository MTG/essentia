/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_DISTRIBUTIONSHAPE_H
#define ESSENTIA_DISTRIBUTIONSHAPE_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class DistributionShape : public Algorithm {

 private:
  Input<std::vector<Real> > _centralMoments;
  Output<Real> _spread;
  Output<Real> _skewness;
  Output<Real> _kurtosis;

 public:
  DistributionShape() {
    declareInput(_centralMoments, "centralMoments", "the central moments of a distribution");
    declareOutput(_spread, "spread", "the spread (variance) of the distribution");
    declareOutput(_skewness, "skewness", "the skewness of the distribution");
    declareOutput(_kurtosis, "kurtosis", "the kurtosis of the distribution");
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

class DistributionShape : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _centralMoments;
  Source<Real> _skewness;
  Source<Real> _spread;
  Source<Real> _kurtosis;

 public:
  DistributionShape() {
    declareAlgorithm("DistributionShape");
    declareInput(_centralMoments, TOKEN, "centralMoments");
    declareOutput(_spread, TOKEN, "spread");
    declareOutput(_skewness, TOKEN, "skewness");
    declareOutput(_kurtosis, TOKEN, "kurtosis");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_DISTRIBUTIONSHAPE_H
