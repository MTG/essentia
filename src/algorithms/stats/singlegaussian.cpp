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

#include "singlegaussian.h"

using namespace TNT;
using namespace JAMA;
using namespace std;
using namespace essentia;
using namespace standard;

const char* SingleGaussian::name = "SingleGaussian";
const char* SingleGaussian::description = DOC("This algorithm implements the single gaussian method. For example, using the single gaussian on descriptors like MFCC with the symmetric Kullback-Leibler divergence might be a much better option than just the mean and variance of the descriptors over a whole signal.\n"
"\n"
"An exception is thrown if the covariance of the input matrix is singular or if the input matrix is empty.\n"
"\n"
"References:\n"
"  [1] E. Pampalk, \"Computational models of music similarity and their\n"
"  application in music information retrieval,” Vienna University of\n"
"  Technology, 2006.");

vector<Real> SingleGaussian::meanMatrix(const Array2D<Real>& matrix, int dim = 1) const {

  int rows = matrix.dim1();
  int columns = matrix.dim2();
  vector<Real> means;

  if (dim == 1) {
    means.resize(columns);
    for (int j=0; j<columns; j++) {
      Real m = 0;
      for (int i=0; i<rows; i++) {
        m += matrix[i][j];
      }
      means[j] = m / rows;
    }
  }
  else {
    if (dim == 2) {
      means.resize(rows);
      for (int i=0; i<rows; i++) {
        Real m = 0;
        for (int j=0; j<columns; j++) {
          m += matrix[i][j];
        }
        means[i]= m / columns;
      }
    }
    else {
      throw EssentiaException("SingleGaussian: The dimension for meanMatrix must be 1 or 2");
    }
  }

  return means;
}

Array2D<Real> SingleGaussian::transposeMatrix(const Array2D<Real>& matrix) const {

  int rows = matrix.dim1();
  int columns = matrix.dim2();
  Array2D<Real> transpose(columns, rows);

  for (int j=0; j<columns; j++) {
    for (int i=0; i<rows; i++) {
      transpose[j][i] = matrix[i][j];
    }
  }

  return transpose;
}


Array2D<Real> SingleGaussian::covarianceMatrix(const Array2D<Real>& matrix, bool lowmem) const {

  int rows = matrix.dim1();
  int columns = matrix.dim2();
  vector<Real> means(columns, 0.0);
  Array2D<Real> cov(columns, columns);

  if (lowmem) {
    // compute means first
    means = meanMatrix(matrix,1);

    // compute covariance matrix
    vector<Real> dim1(rows);

    for (int i=0; i<columns; i++) {
      Real m1 = means[i];
      for (int k=0; k<rows; k++) {
        dim1[k] = matrix[k][i] - m1;
      }
      for (int j=0; j<=i; j++) {
        // compute cov(i,j)
        Real covij = 0.0;
        Real m2 = means[j];
        for (int k=0; k<rows; k++) {
          covij +=  dim1[k] * (matrix[k][j] - m2);
        }
        covij /= (rows - 1); // unbiased estimator
        cov[i][j] = cov[j][i] = covij;
      }
    }
  }
  else { // much faster version, but uses a bit more memory
    // speed optimization: transpose the matrix so that it's in row-major order
    Array2D<Real> transpose = transposeMatrix(matrix);

    // compute means first
    means = meanMatrix(matrix,1);

    // end of optimization: substract means
    for (int i=0; i<columns; i++) {
      for (int j=0; j<rows; j++) {
        transpose[i][j] -= means[i];
      }
    }

    // compute covariance matrix
    for (int i=0; i<columns; i++) {
      for (int j=0; j<=i; j++) {
        // compute cov(i,j)
        Real covij = 0.0;
        for (int k=0; k<rows; k++) {
          covij += transpose[i][k] * transpose[j][k];
        }
        covij /= (rows - 1); // unbiased estimator
        cov[i][j] = cov[j][i] = covij;
      }
    }
  }

  return cov;
}

Array2D<Real> SingleGaussian::inverseMatrix(const Array2D<Real>& matrix) const {
  if (matrix.dim1() != matrix.dim2()) {
    throw EssentiaException("SingleGaussian: Cannot solve linear system because matrix is not a square matrix");
  }

  // make a copy to ensure that the computation of the inverse matrix is done with double precission
  Array2D<double> matrixDouble(matrix.dim1(), matrix.dim2());
  for (int row=0; row<matrix.dim1(); ++row)
    for (int col=0; col<matrix.dim2();++col)
      matrixDouble[row][col] = matrix[row][col];


  LU<double> solver(matrixDouble);
  if (!solver.isNonsingular()) {
    throw EssentiaException("SingleGaussian: Cannot solve linear system because matrix is singular");
  }

  int dim = matrixDouble.dim1();
  Array2D<double> identity(dim, dim, 0.0);
  for (int i=0; i<(int)dim; i++) {
    identity[i][i] = 1.0;
  }

  //return solver.solve(identity);
  Array2D<double> inverseDouble = solver.solve(identity);

  Array2D<Real> inverse(inverseDouble.dim1(), inverseDouble.dim2());
  for (int row=0; row<inverseDouble.dim1(); ++row)
    for (int col=0; col<inverseDouble.dim2();++col)
      inverse[row][col] = inverseDouble[row][col];

  return inverse;
}

void SingleGaussian::compute() {

  const Array2D<Real>& matrix = _matrix.get();

  if (matrix.dim1() == 0 || matrix.dim2() == 0) {
    throw EssentiaException("SingleGaussian: Cannot operate on an empty input matrix");
  }
  if (matrix.dim1() == 1) {
    throw EssentiaException("SingleGaussian: Cannot operate on a matrix with one row");
  }

  vector<Real>& mean = _mean.get();
  Array2D<Real>& covariance = _covariance.get();
  Array2D<Real>& inverseCovariance = _inverseCovariance.get();

  mean = meanMatrix(matrix,1);

  covariance = covarianceMatrix(matrix, false);

  inverseCovariance = inverseMatrix(covariance);
}
