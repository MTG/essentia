/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_SPLINE_H
#define ESSENTIA_SPLINE_H

#include "algorithm.h"
#include "splineutil.h"

namespace essentia {
namespace standard {

class Spline : public Algorithm {

  enum splineType { B, BETA, QUADRATIC };

 protected:
  Input<Real> _xInput;
  Output<Real> _yOutput;

  std::vector<double> _xPoints;
  std::vector<double> _yPoints;
  double _beta1, _beta2;
  splineType _type;

 public:
  Spline() {
    declareInput(_xInput, "x", "the input coordinate (x-axis)");
    declareOutput(_yOutput, "y", "the value of the spline at x");
  }

  ~Spline() {}

  void declareParameters() {
    std::vector<Real> defaultPoints(2);
    defaultPoints[0] = 0;
    defaultPoints[1] = 1;
    declareParameter("xPoints", "the x-coordinates where data is specified (the points must be arranged in ascending order and cannot contain duplicates)", "", defaultPoints);
    declareParameter("yPoints", "the y-coordinates to be interpolated (i.e. the known data)", "", defaultPoints);
    declareParameter("type", "the type of spline to be computed", "{b,beta,quadratic}", "b");
    declareParameter("beta1", "the skew or bias parameter (only available for type beta)", "[0,inf]", 1.0);
    declareParameter("beta2", "the tension parameter", "[0,inf)", 0.0);
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

class Spline : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _xInput;
  Source<Real> _yOutput;

 public:
  Spline() {
    declareAlgorithm("Spline");
    declareInput(_xInput, TOKEN, "x");
    declareOutput(_yOutput, TOKEN, "y");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_SPLINE_H
