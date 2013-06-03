/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
