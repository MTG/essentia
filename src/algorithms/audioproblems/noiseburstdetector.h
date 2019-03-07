/*
 * Copyright (C) 2006-2019  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_NOISEBURSTDETECTOR_H
#define ESSENTIA_NOISEBURSTDETECTOR_H

#include "algorithm.h"
#include "algorithmfactory.h"
#include "essentiamath.h"

namespace essentia {
namespace standard {

class NoiseBurstDetector : public Algorithm {
 private:
  Input<std::vector<Real>> _frame;
  Output<std::vector<Real>> _indexes;

  Real _threshold;
  Real _thresholdCoeff;
  Real _silenceThreshold;
  Real _alpha;

  Algorithm* _Clipper;

  Real robustRMS(std::vector<Real> x, Real k);
  void updateEMA(Real x);

 public:
  NoiseBurstDetector() {
    declareInput(_frame, "frame", "the input frame (must be non-empty)");
    declareOutput(_indexes, "indexes", "indexes of the noisy samples");

    _Clipper = AlgorithmFactory::create("Clipper");
  }

  ~NoiseBurstDetector() {
    if (_Clipper) delete _Clipper;
  }

  void declareParameters() {
    declareParameter("threshold", "factor to control the dynamic theshold", "(-inf,inf)", 8);
    declareParameter("silenceThreshold", "threshold to skip silent frames", "(-inf,0)", -50);
    declareParameter("alpha", "alpha coefficient for the Exponential Moving Average threshold estimation.", "(0,1)", .9);
  }

  void configure();
  void compute();
  
  static const char *name;
  static const char *category;
  static const char *description;
};

} // namespace standard
} // namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class NoiseBurstDetector : public StreamingAlgorithmWrapper {
 protected:
  Sink<std::vector<Real>> _frame;
  Source<std::vector<Real>> _indexes;

 public:
  NoiseBurstDetector() {
    declareAlgorithm("NoiseBurstDetector");
    declareInput(_frame, TOKEN, "frame");
    declareOutput(_indexes, TOKEN, "indexes");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_NOISEBURSTDETECTOR_H
