/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of Essentia
 *
 * Essentia is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
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
