/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
