/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "cubicspline.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* CubicSpline::name = "CubicSpline";
const char* CubicSpline::description = DOC("Computes the second derivatives of a piecewise cubic spline.\n"
"The input value, i.e. the point at which the spline is to be evaluated typically should be between xPoints[0] and xPoints[size-1]. If the value lies outside this range, extrapolation is used."
"\n"
"Regarding [left/right] boundary condition flag parameters:\n"
"  - 0: the cubic spline should be a quadratic over the first interval\n"
"  - 1: the first derivative at the [left/right] endpoint should be [left/right]BoundaryFlag\n"
"  - 2: the second derivative at the [left/right] endpoint should be [left/right]BoundaryFlag\n"
"References:\n"
"  [1] Spline interpolation - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Spline_interpolation");

void CubicSpline::compute() {
  const double& xInput = _xInput.get();
  Real& yOutput = _yOutput.get();
  Real& dyOutput = _dyOutput.get();
  Real& ddyOutput = _ddyOutput.get();

  double dy = 0.0, ddy = 0.0;

  yOutput = (Real)spline_cubic_val((int)_xPoints.size(), &_xPoints[0], xInput, &_yPoints[0],
                                   _splineSecondDerivatives, &dy, &ddy);
  dyOutput = dy;
  ddyOutput = ddy;
}


void CubicSpline::configure() {
  vector<Real> x = parameter("xPoints").toVectorReal();
  vector<Real> y = parameter("yPoints").toVectorReal();
  if (x.size() != y.size() ) {
    throw EssentiaException("CubicSpline: parameter 'xPoints' must have the same size than parameter 'yPoints')");
  }
  int size = x.size();
  for (int i=0; i<size-1; ++i) {
    if (x[i]>=x[i+1]) {
      throw EssentiaException("CubicSpline: parameter 'xPoints' must be in ascendant order and cannot contain duplicates)");
    }
  }
  _xPoints.resize(size);
  _yPoints.resize(size);
  for (int i=0; i<size; ++i) {
    _xPoints[i] = double(x[i]);
    _yPoints[i] = double(y[i]);
  }

  _leftBoundaryFlag = parameter("leftBoundaryFlag").toInt();
  _leftBoundaryValue = parameter("leftBoundaryValue").toReal();
  _rightBoundaryFlag = parameter("rightBoundaryFlag").toInt();
  _rightBoundaryValue = parameter("rightBoundaryValue").toReal();
  _splineSecondDerivatives = spline_cubic_set(int(_xPoints.size()), &_xPoints[0], &_yPoints[0],
                                              _leftBoundaryFlag,  _leftBoundaryValue,
                                              _rightBoundaryFlag, _rightBoundaryValue);
}
