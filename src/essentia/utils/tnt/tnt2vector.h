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
