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

#ifndef ESSENTIA_WARPEDAUTOCORRELATION_H
#define ESSENTIA_WARPEDAUTOCORRELATION_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class WarpedAutoCorrelation : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _warpedAutoCorrelation;

 public:
  WarpedAutoCorrelation() {
    declareInput(_signal, "array", "the array to be analyzed");
    declareOutput(_warpedAutoCorrelation, "warpedAutoCorrelation", "the warped auto-correlation vector");
  }

  void declareParameters() {
    declareParameter("maxLag", "the maximum lag for which the auto-correlation is computed (inclusive) (must be smaller than signal size) ", "(0,inf)", 1);
    declareParameter("sampleRate", "the audio sampling rate [Hz]", "(0,inf)", 44100.);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

 private:
  Real _lambda;
  std::vector<Real> _tmp;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class WarpedAutoCorrelation : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<std::vector<Real> > _warpedAutoCorrelation;

 public:
  WarpedAutoCorrelation() {
    declareAlgorithm("WarpedAutoCorrelation");
    declareInput(_signal, TOKEN, "array");
    declareOutput(_warpedAutoCorrelation, TOKEN, "warpedAutoCorrelation");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_WARPEDAUTOCORRELATION_H
