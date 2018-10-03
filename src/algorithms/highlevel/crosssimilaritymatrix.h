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
#include "utils/tnt/tnt.h"
#include "utils/tnt/tnt2essentiautils.h"
#include <complex>
namespace essentia {
namespace standard {
 class CrossSimilarityMatrix : public Algorithm {
  protected:
  Input<std::vector<std::vector<Real>> > _queryFeature;
  Input<std::vector<std::vector<Real>> > _referenceFeature;
  Output<std::vector<std::vector<Real>> > _csm;
  int tau;
  int m;
  double kappa;
  int noti;
  bool oti;
  bool toBlocked;
  public:
  CrossSimilarityMatrix() {
    declareInput(_queryFeature, "queryFeature", " audio chromafeature vector of the query song as input");
    declareInput(_referenceFeature, "referenceFeature", "audio chroma feature vector of the reference song as input");
    declareOutput(_csm, "csm", "2d cross similarity matrix of the query and reference song audio features");
   }

   void declareParameters() {
    declareParameter("tau", "time delay for embedding in units of number of windows", "[1,inf)", 1);
    declareParameter("m", "embedding dimension for the stacked feature embedding", "[0,inf)", 9);
    declareParameter("kappa", "fraction of mutual nearest neighbours to consider", "[0,1]", 0.095);
    declareParameter("oti", "whether to transpose the key of the reference song to the query song by (OTI)", "{true,false}", false);
    declareParameter("toBlocked", "whether to use stacked chroma vector embedding for computing similarity", "{true,false}", false);
    declareParameter("noti", "Number of circular shifts to be checked for optimal transposition index", "[0, inf)", 12);
  }

  void configure();
  void compute();
  std::vector<Real> globalAverageChroma(std::vector<std::vector<Real> >& inputFeature) const;
  std::vector<std::vector<Real> > toTimeEmbedding(std::vector<std::vector<Real> >& inputArray, int m, int tau) const;
  int optimalTranspositionIndex(std::vector<std::vector<Real> >& featureA, std::vector<std::vector<Real> >& featureB, int nshifts) const;
  static const char* name;
  static const char* category;
  static const char* description;

  int _tau;
  int _m;
  double _kappa;
  int _noti;
  bool _oti;
  bool _toBlocked;

  /*protected:
   std::vector<Real> globalAverageChroma(std::vector<std::vector<Real> >& inputFeature) const;
   std::vector<std::vector<Real> > toTimeEmbedding(std::vector<std::vector<Real> >& inputArray, int m, int tau) const;
   int optimalTranspositionIndex(std::vector<std::vector<Real> >& featureA, std::vector<std::vector<Real> >& featureB, int nshifts) const;
  */
 };
 } // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"
namespace essentia {
namespace streaming {
 class CrossSimilarityMatrix : public StreamingAlgorithmWrapper {
  protected:
  Sink<std::vector<Real> > _queryFeature;
  Sink<std::vector<Real> > _referenceFeature;
  Source<std::vector<std::vector<Real>> > _csm;
  public:
  CrossSimilarityMatrix() {
    declareAlgorithm("CrossSimilarityMatrix");
    declareInput(_queryFeature, TOKEN, "queryFeature");
    declareInput(_referenceFeature, TOKEN, "referenceFeature");
    declareOutput(_csm, TOKEN, "csm");
  }
};
} // namespace streaming
} // namespace essentia
 #endif // ESSENTIA_CrossSimilarityMatrix_H
