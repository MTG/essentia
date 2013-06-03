#ifndef ESSENTIA_TNT2ESSENTIAUTILS_H
#define ESSENTIA_TNT2ESSENTIAUTILS_H

#include <fstream>
#include "tnt.h"

namespace essentia {

  template <typename T>
  TNT::Array2D<T>&  operator/=(TNT::Array2D<T> &A, const T &k) {
    int m = A.dim1();
    int n = A.dim2();

    if (k == 0) {
      throw EssentiaException("Error: Division of an TNT::Array2D by zero");
    }

    for (int i=0; i<m; i++)
      for (int j=0; j<n; j++)
        A[i][j] /= k;
    return A;
  }

  template <typename T>
  TNT::Array2D<T>  operator/(const TNT::Array2D<T> &A, const T &k) {
    int m = A.dim1();
    int n = A.dim2();

    if (k == 0) {
      throw EssentiaException("Error: Division of an TNT::Array2D by zero");
    }

    if (m == 0 || n == 0 )
      return TNT::Array2D<T>();
    TNT::Array2D<T> result(m,n);

    for (int i=0; i<m; i++)
      for (int j=0; j<n; j++)
        result[i][j] = A[i][j] / k;

    return result;
  }

  template <typename T>
  TNT::Array2D<T>& matinit(TNT::Array2D<T>& A) {
    for (int i = 0; i < A.dim1(); i++) {
      for (int j = 0; j < A.dim2(); j++) {
        A[i][j] = 0;
      }
    }
    return A;
  }

} // namespace essentia

#endif // ESSENTIA_TNT2ESSENTIAUTILS_H
