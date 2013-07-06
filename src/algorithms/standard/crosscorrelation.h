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
