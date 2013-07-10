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

#ifndef ESSENTIA_CUBIC_SPLINE_H
#define ESSENTIA_CUBIC_SPLINE_H

#include "algorithm.h"
#include "splineutil.h"

namespace essentia {
namespace standard {

class CubicSpline : public Algorithm {

 protected:
  Input<Real> _xInput;
  Output<Real> _yOutput;
  Output<Real> _dyOutput;
  Output<Real> _ddyOutput;

  std::vector<double> _xPoints;
  std::vector<double> _yPoints;
  int _leftBoundaryFlag;
  int _rightBoundaryFlag;
  double _leftBoundaryValue;
  double _rightBoundaryValue;
  double * _splineSecondDerivatives;

 public:
  CubicSpline() {
    declareInput(_xInput, "x", "the input coordinate (x-axis)");
    declareOutput(_yOutput, "y", "the value of the spline at x");
    declareOutput(_dyOutput, "dy", "the first derivative of the spline at x");
    declareOutput(_ddyOutput, "ddy", "the second derivative of the spline at x");
  }

  ~CubicSpline() {}

  void declareParameters() {
    std::vector<Real> defaultPoints(2);
    defaultPoints[0] = 0;
    defaultPoints[1] = 1;
    declareParameter("xPoints", "the x-coordinates where data is specified (the points must be arranged in ascending order and cannot contain duplicates)", "", defaultPoints);
    declareParameter("yPoints", "the y-coordinates to be interpolated (i.e. the known data)", "", defaultPoints);
    declareParameter("leftBoundaryFlag", "type of boundary condition for the left boundary", "{0,1,2}", 0);
    declareParameter("leftBoundaryValue", "the value to be used in the left boundary, when leftBoundaryFlag is 1 or 2", "(-inf,inf)", 0.0);
    declareParameter("rightBoundaryFlag", "type of boundary condition for the right boundary", "{0,1,2}", 0);
    declareParameter("rightBoundaryValue", "the value to be used in the right boundary, when rightBoundaryFlag is 1 or 2", "(-inf,inf)", 0.0);
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

class CubicSpline : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _xInput;
  Source<Real> _yOutput;
  Source<Real> _dyOutput;
  Source<Real> _ddyOutput;

 public:
  CubicSpline() {
    declareAlgorithm("CubicSpline");
    declareInput(_xInput, TOKEN, "x");
    declareOutput(_yOutput, TOKEN, "y");
    declareOutput(_dyOutput, TOKEN, "dy");
    declareOutput(_ddyOutput, TOKEN, "ddy");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_CUBIC_SPLINE_H
