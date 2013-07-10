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

#ifndef ESSENTIA_PCA_H
#define ESSENTIA_PCA_H

#include "algorithm.h"
#include "pool.h"

namespace essentia {
namespace standard {

class PCA : public Algorithm {

 protected:
  Input<Pool> _poolIn;
  Output<Pool> _poolOut;

 public:
  PCA() {
    declareInput(_poolIn, "poolIn", "the pool where to get the spectral contrast feature vectors");
    declareOutput(_poolOut, "poolOut", "the pool where to store the transformed feature vectors");
  }

  ~PCA(){}

  void declareParameters() {
    declareParameter("namespaceIn", "will look for this namespace in poolIn", "", "spectral contrast");
    declareParameter("namespaceOut", "will save to this namespace in poolOut", "", "spectral contrast pca");
    declareParameter("dimensions", "number of dimension to reduce the input to", "[0, inf)", 0);
  }

  void configure(){}
  void compute();

  static const char* name;
  static const char* description;

};

} //namespace standard
} //namespace essentia


#endif // ESSENTIA_PCA_H
