#ifndef ESSENTIA_TNT2VECTOR_H
#define ESSENTIA_TNT2VECTOR_H

#include <fstream>
#include "tnt.h"

namespace essentia {

  inline TNT::Array2D<Real> vecvecToArray2D(const std::vector<std::vector<Real> >& v) {

    if (v.size() == 0) {
      throw EssentiaException("You are trying to convert an empty vector of vector into a Array2D.");
    }

    TNT::Array2D<Real> v2D((int)v.size(), (int)v[0].size());
    for (int i=0; i<v2D.dim1(); i++) {
      for (int j=0; j<v2D.dim2(); j++) {
        v2D[i][j] = v[i][j];
      }
    }

    return v2D;
  }

  inline std::vector<std::vector<Real> > array2DToVecvec(const TNT::Array2D<Real>& v2D) {

    if (v2D.dim1() == 0) {
      throw EssentiaException("You are trying to convert an empty Array2D into a vector of vector."); 
    }

    std::vector<std::vector<Real> > v;
    v.resize(v2D.dim1());

    for (uint i=0; i<v.size(); i++) {
      v[i].resize(v2D.dim2());
      for (uint j=0; j<v[i].size(); j++) {
        v[i][j] = v2D[(int)i][(int)j];
      }
    }

    return v;
  }

} // namespace essentia

#endif // ESSENTIA_TNT2VECTOR_H
