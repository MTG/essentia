/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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
#ifndef ESSENTIA_CROSSSIMILARITYMATRIX_H
#define ESSENTIA_CROSSSIMILARITYMATRIX_H
#include "algorithmfactory.h"
#include <complex>

namespace essentia {
namespace standard {

class CrossSimilarityMatrix : public Algorithm {
  protected:
   Input<std::vector<std::vector<Real> > > _queryFeature;
   Input<std::vector<std::vector<Real> > > _referenceFeature;
   Output<std::vector<std::vector<Real> > > _csm;
  public:
   CrossSimilarityMatrix() {
    declareInput(_queryFeature, "queryFeature", "input chromagram of the query song");
    declareInput(_referenceFeature, "referenceFeature", "input chromagram of the reference song");
    declareOutput(_csm, "csm", "2D binary cross-similarity matrix of the query and reference features");
   }

   void declareParameters() {
    declareParameter("tau", "time delay for embedding in units of number of windows", "[1,inf)", 1);
    declareParameter("embedDimension", "embedding dimension for the stacked feature embedding. Choose 'embedDimension=1' to use raw input feature vector for the similarity calculation.", "[0,inf)", 9);
    declareParameter("kappa", "fraction of mutual nearest neighbours to consider while computing euclidean distances", "[0,1]", 0.095);
    declareParameter("oti", "whether to transpose the key of the reference song to the query song by (OTI)", "{true,false}", true);
    declareParameter("toBlocked", "whether to use stacked chroma vector embedding for computing similarity", "{true,false}", true);
    declareParameter("noti", "Number of circular shifts to be checked for optimal transposition index", "[0, inf)", 12);
    declareParameter("otiBinary", "whether to use the OTI-based chroma binary similarity method", "{true,false}", false);
    declareParameter("optimiseThreshold", "whether to use the optimised threhold method in hte similarity computation ", "{true,false}", false);
  }

   void configure();
   void compute();

   static const char* name;
   static const char* category;
   static const char* description;

  protected:
   int _tau;
   int _embedDimension;
   Real _kappa;
   int _noti;
   bool _oti;
   bool _toBlocked;
   bool _otiBinary;
   bool _optimiseThreshold;
   Real _mathcCoef;
   Real _mismatchCoef;

};

} // namespace standard
} // namespace essentia

#include "pool.h"
#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

class CrossSimilarityMatrix : public Algorithm {
 protected:
  Sink<std::vector<Real> > _queryFeature;
  Source<TNT::Array2D<Real> > _csm;

  // params variables
  int _tau;
  int _embedDimension;
  Real _kappa;
  int _noti;
  bool _oti;
  bool _otiBinary;
  Real _mathcCoef;
  Real _mismatchCoef;
  Real _minFramesSize;
  std::vector<std::vector<Real> > _prevQueryFrame;
  std::vector<std::vector<Real> > _referenceFeature;

  std::vector<std::vector<Real> > streamingFrames2TimeEmbedding(std::vector<std::vector<Real> > inputFrames, int m, int tau);

 public:
  CrossSimilarityMatrix() : Algorithm() {
    declareInput(_queryFeature, 10, "queryFeature", "input chromagram of the query song");
    declareOutput(_csm, 1, "csm", "2D binary cross-similarity matrix of the query and reference features");
  }

  ~CrossSimilarityMatrix() {}

  AlgorithmStatus process();

  void declareParameters() {
    declareParameter("referenceFeature", "input chromagram of the reference song", "", std::vector<std::vector<Real> >());
    declareParameter("tau", "time delay for embedding in units of number of windows", "[1,inf)", 1);
    declareParameter("embedDimension", "embedding dimension for the stacked feature embedding. Choose embedDimension=1 to use raw input feature vector for the similarity calculation.", "[0,inf)", 9);
    declareParameter("kappa", "fraction of mutual nearest neighbours to consider while computing euclidean distances", "[0,1]", 0.095);
    declareParameter("oti", "whether to transpose the key of the reference song to the query song by (OTI)", "{true,false}", true);
    declareParameter("noti", "Number of circular shifts to be checked for optimal transposition index", "[0, inf)", 12);
    declareParameter("otiBinary", "whether to use the OTI-based chroma binary similarity method", "{true,false}", false);
  }

  void configure();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace streaming
} // namespace essentia
 #endif // ESSENTIA_CROSSSIMILARITYMATRIX_H
