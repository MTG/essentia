/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_REALACCUMULATOR_H
#define ESSENTIA_REALACCUMULATOR_H

#include "streamingalgorithmcomposite.h"
#include "vectoroutput.h"

namespace essentia {
namespace streaming {

class RealAccumulator : public AlgorithmComposite {
 protected:
  SinkProxy<Real> _value;
  Source<std::vector<Real> > _array;
  std::vector<Real> _accu;
  Algorithm* _vectorOutput;

 public:
  RealAccumulator();
  ~RealAccumulator();

  void declareParameters() {}

  void reset();
  AlgorithmStatus process();

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_vectorOutput));
    declareProcessStep(SingleShot(this));
  }

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_REALACCUMULATOR_H
