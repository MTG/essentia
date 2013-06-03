/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_CROSSCORRELATION_H
#define ESSENTIA_CROSSCORRELATION_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class CrossCorrelation : public Algorithm {

 private:
  Input<std::vector<Real> > _signal_x;
  Input<std::vector<Real> > _signal_y;
  Output<std::vector<Real> > _correlation;

 public:
  CrossCorrelation() {
    declareInput(_signal_x, "arrayX", "the first input array");
    declareInput(_signal_y, "arrayY", "the second input array");
    declareOutput(_correlation, "crossCorrelation", "the cross-correlation vector between the two input arrays (its size is equal to maxLag - minLag + 1)");
  }

  void declareParameters() {
    declareParameter("minLag", "the minimum lag to be computed between the two vectors", "(-inf,inf)", 0);
    declareParameter("maxLag", "the maximum lag to be computed between the two vectors", "(-inf,inf)", 1);
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

class CrossCorrelation : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal_y;
  Sink<std::vector<Real> > _signal_x;
  Source<std::vector<Real> > _correlation;

 public:
  CrossCorrelation() {
    declareAlgorithm("CrossCorrelation");
    declareInput(_signal_x, TOKEN, "arrayX");
    declareInput(_signal_y, TOKEN, "arrayY");
    declareOutput(_correlation, TOKEN, "crossCorrelation");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_CROSSCORRELATION_H
