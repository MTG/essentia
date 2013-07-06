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
