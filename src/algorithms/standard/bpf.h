/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_BPF_H
#define ESSENTIA_BPF_H

#include "algorithm.h"
#include "bpfutil.h"

namespace essentia {
namespace standard {

class BPF : public Algorithm {

 protected:
  Input<Real> _xInput;
  Output<Real> _yOutput;

  essentia::util::BPF bpf;
 public:
  BPF() {
    declareInput(_xInput, "x", "the input coordinate (x-axis)");
    declareOutput(_yOutput, "y", "the output coordinate (y-axis)");
  }

  ~BPF() {}

  void declareParameters() {
    std::vector<Real> defaultPoints(2);
    defaultPoints[0] = 0;
    defaultPoints[1] = 1;
    declareParameter("xPoints", "the x-coordinates of the points forming the break-point function (the points must be arranged in ascending order and cannot contain duplicates)", "", defaultPoints);
    declareParameter("yPoints", "the y-coordinates of the points forming the break-point function", "", defaultPoints);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;
};

} // namespace essentia
} // namespace standard

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class BPF : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _xInput;
  Source<Real> _yOutput;

 public:
  BPF() {
    declareAlgorithm("BPF");
    declareInput(_xInput, TOKEN, "x");
    declareOutput(_yOutput, TOKEN, "y");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_BPF_H
