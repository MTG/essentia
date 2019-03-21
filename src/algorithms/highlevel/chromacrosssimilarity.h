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
#ifndef ESSENTIA_CHROMACROSSSIMILARITY_H
#define ESSENTIA_CHROMACROSSSIMILARITY_H
#include "algorithmfactory.h"
#include <complex>

namespace essentia {
namespace standard {

class ChromaCrossSimilarity : public Algorithm {
  protected:
   Input<std::vector<std::vector<Real> > > _queryFeature;
   Input<std::vector<std::vector<Real> > > _referenceFeature;
   Output<std::vector<std::vector<Real> > > _csm;
  public:
   ChromaCrossSimilarity() {
    declareInput(_queryFeature, "queryFeature", "input chromagram of the query song (e.g., a HPCP)");
    declareInput(_referenceFeature, "referenceFeature", "input chromagram of the reference song (e.g., a HPCP)");
    declareOutput(_csm, "csm", "2D binary cross-similarity matrix of the query and reference features");
   }

   void declareParameters() {
    declareParameter("frameStackStride", "time delay for stacking the features in the units of number of frames", "[1,inf)", 1);
    declareParameter("frameStackSize", "number of input frames to stack together and treat as a feature vector for similarity computation. Choose 'frameStackSize=1' to use the original input frames without stacking", "[0,inf)", 9);
    declareParameter("binarizePercentile", "maximum percent of distance values to consider as similar in each row and each column", "[0,1]", 0.095);
    declareParameter("oti", "whether to transpose the key of the reference song to the query song by (OTI)", "{true,false}", true);
    declareParameter("noti", "Number of circular shifts to be checked for optimal transposition index", "[0, inf)", 12);
    declareParameter("otiBinary", "whether to use the OTI-based chroma binary similarity method", "{true,false}", false);
    declareParameter("optimizeThreshold", "whether to use the optimised threhold method in the similarity computation (While enalbled it outputs same as in essentia streaming mode 'ChromaCrossSimilarity' algorithm)", "{true,false}", false);
  }

   void configure();
   void compute();

   static const char* name;
   static const char* category;
   static const char* description;

  protected:

   int _frameStackStride;
   int _frameStackSize;
   Real _binarizePercentile;
   int _noti;
   bool _oti;
   bool _otiBinary;
   bool _optimizeThreshold;
   Real _mathcCoef;
   Real _mismatchCoef;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

class ChromaCrossSimilarity : public Algorithm {
 protected:
  Sink<std::vector<Real> > _queryFeature;
  Source<std::vector<Real> > _csm;

  // params variables
  int _frameStackStride;
  int _frameStackSize;
  Real _binarizePercentile;
  int _noti;
  bool _oti;
  bool _otiBinary;
  Real _mathcCoef;
  Real _mismatchCoef;
  Real _minFramesSize;
  std::vector<std::vector<Real> > _referenceFeature;
  std::vector<std::vector<Real> > _referenceFeatureStack;
  std::vector<std::vector<Real> > _outputSimMatrix;

 public:
  ChromaCrossSimilarity() : Algorithm() {
    declareInput(_queryFeature, 10, "queryFeature", "input chromagram of the query song. (eg: a HPCP)");
    declareOutput(_csm, 1, "csm", "2D binary cross-similarity matrix of the query and reference chromagram");
  }

  ~ChromaCrossSimilarity() {}

  AlgorithmStatus process();

  void declareParameters() {
    declareParameter("referenceFeature", "input chromagram of the reference song. (eg. a HPCP)", "", std::vector<std::vector<Real> >());
    declareParameter("frameStackStride", "stride size to form a stack of frames (e.g., 'frameStackStride'=1 to use consecutive frames; 'frameStackStride'=2 for using every second frame)", "[1,inf)", 1);
    declareParameter("frameStackSize", "embedding dimension for the stacked feature embedding. Choose embedDimension=1 to use raw input feature vector for the similarity calculation.", "[0,inf)", 9);
    declareParameter("binarizePercentile", "maximum percent of distance values to consider as similar in each row and each column", "[0,1]", 0.095);
    declareParameter("oti", "optimal transposition index of the query and reference song", "[0, inf]", 0);
    declareParameter("otiBinary", "whether to use the OTI-based chroma binary similarity method", "{true,false}", false);
  }

  void configure();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace streaming
} // namespace essentia
 #endif // ESSENTIA_CHROMACROSSSIMILARITY_H
