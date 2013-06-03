/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_SILENCERATE_H
#define ESSENTIA_SILENCERATE_H


#include "algorithm.h"

namespace essentia {
namespace standard {

class SilenceRate : public Algorithm {

 protected:
  Input<std::vector<Real> > _frame;
  std::vector<Output<Real>*> _outputs;

  std::vector<Real> _thresholds;

 public:
  SilenceRate() {
    declareInput(_frame, "frame", "the input frame");
  }

  ~SilenceRate() {}

  void declareParameters() {
    declareParameter("thresholds", "the threshold values", "", std::vector<Real>());
  }

  void configure();

  void compute();
  void clearOutputs();
  void reset() {}

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

/**
 * @todo make sure this algo doesn't break when configuring more than once.
 *       (it most certainly does, right now)
 */
class SilenceRate : public Algorithm {
 protected:
  Sink<std::vector<Real> > _frame;
  std::vector<Source<Real>*> _outputs;

  std::vector<Real> _thresholds;

  void clearOutputs();

 public:

  SilenceRate() : Algorithm() {
    declareInput(_frame, 1, "frame", "the input frame");
  }

  ~SilenceRate() {
    clearOutputs();
  }

  void declareParameters() {
    declareParameter("thresholds", "the threshold values", "", std::vector<Real>());
  }

  void configure();

  AlgorithmStatus process();

  static const char* name;
  static const char* description;

};


} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_SILENCERATE_H
