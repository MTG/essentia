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

#ifndef ESSENTIA_SINGLEGAUSSIAN_H
#define ESSENTIA_SINGLEGAUSSIAN_H

#include "algorithm.h"
#include "tnt/tnt.h"
#include "tnt/jama_lu.h"

namespace essentia {
namespace standard {

class SingleGaussian : public Algorithm {

 private:
  Input<TNT::Array2D<Real> > _matrix;
  Output<std::vector<Real> > _mean;
  Output<TNT::Array2D<Real> > _covariance;
  Output<TNT::Array2D<Real> > _inverseCovariance;

 public:
  SingleGaussian() {
    declareInput(_matrix, "matrix", "the input data matrix (e.g. the MFCC descriptor over frames)");
    declareOutput(_mean, "mean", "the mean of the values");
    declareOutput(_covariance, "covariance", "the covariance matrix");
    declareOutput(_inverseCovariance, "inverseCovariance", "the inverse of the covariance matrix");
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;

 protected:
  std::vector<Real> meanMatrix(const TNT::Array2D<Real>& matrix, int dim) const;
  TNT::Array2D<Real> transposeMatrix(const TNT::Array2D<Real>& matrix) const;
  TNT::Array2D<Real> covarianceMatrix(const TNT::Array2D<Real>& matrix, bool lowmem = false) const;
  TNT::Array2D<Real> inverseMatrix(const TNT::Array2D<Real>& matrix) const;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class SingleGaussian : public StreamingAlgorithmWrapper {

 protected:
  Sink<TNT::Array2D<Real> > _matrix;
  Source<std::vector<Real> > _mean;
  Source<TNT::Array2D<Real> > _covariance;
  Source<TNT::Array2D<Real> > _inverseCovariance;

 public:
  SingleGaussian() {
    declareAlgorithm("SingleGaussian");
    declareInput(_matrix, TOKEN, "matrix");
    declareOutput(_mean, TOKEN, "mean");
    declareOutput(_covariance, TOKEN, "covariance");
    declareOutput(_inverseCovariance, TOKEN, "inverseCovariance");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_SINGLEGAUSSIAN_H
