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

#include "pca.h"
#include "algorithmfactory.h"
#include "tnt/tnt2vector.h"
#include "tnt/jama_lu.h"
#include "tnt/jama_eig.h"

using namespace std;
using namespace TNT;
using namespace JAMA;
using namespace essentia;
using namespace standard;

const char* PCA::name = "PCA";
const char* PCA::description = DOC("Karhunen Loeve Transform || Principal Component Analysis based on the covariance matrix of the signal.\n"
"\n"
"References:\n"
"  [1] Principal component analysis - Wikipedia, the free enciclopedia\n"
"  http://en.wikipedia.org/wiki/Principal_component_analysis");

void PCA::compute() {

  const Pool& poolIn = _poolIn.get();
  Pool& poolOut = _poolOut.get();

  // get data from the pool
  string nameIn = parameter("namespaceIn").toString();
  string nameOut = parameter("namespaceOut").toString();
  vector<vector<Real> > rawFeats = poolIn.value<vector<vector<Real> > >(nameIn);

  // how many dimensions are there?
  int bands = rawFeats[0].size();

  // calculate covariance for this songs frames
  // before there was an implementation for covariance from Vincent akkerman. I (eaylon) think it is better
  // and more maintainable to use an algorithm that computes covariance.
  // Using singleGaussian algo seems to give slightly different results for variances (8 decimal places)
  Array2D<Real> matrix, covMatrix, icov;
  vector<Real> means;
  matrix = vecvecToArray2D(rawFeats);
  Algorithm* sg = AlgorithmFactory::create("SingleGaussian");
  sg->input("matrix").set(matrix);
  sg->output("mean").set(means);
  sg->output("covariance").set(covMatrix);
  sg->output("inverseCovariance").set(icov);
  sg->compute();
  delete sg;

  // calculate eigenvectors, get the eigenvector matrix
  Eigenvalue<Real> eigMatrixCalc(covMatrix);
  Array2D<Real>    eigMatrix;
  eigMatrixCalc.getV(eigMatrix);

  int nFrames = rawFeats.size();
  for (int row=0; row<nFrames; row++) {
    for (int col=0; col<bands; col++) {
      rawFeats[row][col] -= means[col];
    }
  }

  // reduce dimensions of eigMatrix
  int requiredDimensions = parameter("dimensions").toInt();
  if (requiredDimensions > eigMatrix.dim2() || requiredDimensions < 1)
    requiredDimensions = eigMatrix.dim2();
  Array2D<Real> reducedEig(eigMatrix.dim1(), requiredDimensions);

  for (int row=0; row<eigMatrix.dim1(); row++) {
    for (int column=0; column<requiredDimensions; column++) {
      reducedEig[row][column] = eigMatrix[row][column+eigMatrix.dim2()-requiredDimensions];
    }
  }

  // transform all the frames and add to the output
  Array2D<Real> featVector(1,bands, 0.0);
  vector<Real> results = vector<Real>(requiredDimensions, 0.0);
  for (int row=0; row<nFrames; row++) {
    for (int col=0; col<bands; col++) {
      featVector[0][col] = rawFeats[row][col];
    }
    featVector = matmult(featVector, reducedEig);
    for (int i=0; i<requiredDimensions; i++) {
      results[i] = featVector[0][i];
    }
    poolOut.add(nameOut, results);
  }
}
