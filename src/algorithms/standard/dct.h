/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_DCT_H
#define ESSENTIA_DCT_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class DCT : public Algorithm {

 protected:
  Input<std::vector<Real> > _array;
  Output<std::vector<Real> > _dct;

 public:
  DCT() {
    declareInput(_array, "array", "the input array");
    declareOutput(_dct, "dct", "the discrete cosine transform of the input array");
  }

  void declareParameters() {
    declareParameter("inputSize", "the size of the input array", "[1,inf)", 10);
    declareParameter("outputSize", "the number of output coefficients", "[1,inf)", 10);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;


 protected:
  int _outputSize;
  void createDctTable(int inputSize, int outputSize);

  std::vector<std::vector<Real> > _dctTable;
};

} // namespace essentia
} // namespace standard


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class DCT : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<std::vector<Real> > _dct;

 public:
  DCT() {
    declareAlgorithm("DCT");
    declareInput(_array, TOKEN, "array");
    declareOutput(_dct, TOKEN, "dct");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_DCT_H
