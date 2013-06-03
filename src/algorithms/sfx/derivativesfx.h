/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_DERIVATIVESFX_H
#define ESSENTIA_DERIVATIVESFX_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class DerivativeSFX : public Algorithm {

 protected:
  Input<std::vector<Real> > _envelope;
  Output<Real> _derAvAfterMax;
  Output<Real> _maxDerBeforeMax;

 public:
  DerivativeSFX() {
    declareInput(_envelope, "envelope", "the envelope of the signal");
    declareOutput(_derAvAfterMax, "derAvAfterMax", "the weighted average of the derivative after the maximum amplitude");
    declareOutput(_maxDerBeforeMax, "maxDerBeforeMax", "the maximum derivative before the maximum amplitude");
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

class DerivativeSFX : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _envelope;
  Source<Real> _derAvAfterMax;
  Source<Real> _maxDerBeforeMax;

 public:
  DerivativeSFX() {
    declareAlgorithm("DerivativeSFX");
    declareInput(_envelope, TOKEN, "envelope");
    declareOutput(_derAvAfterMax, TOKEN, "derAvAfterMax");
    declareOutput(_maxDerBeforeMax, TOKEN, "maxDerBeforeMax");
  }

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_DERIVATIVESFX_H
