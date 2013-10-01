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

#include "spline.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Spline::name = "Spline";
const char* Spline::description = DOC("Evaluates a piecewise spline of type b, beta or quadratic.\n"
"The input value, i.e. the point at which the spline is to be evaluated typically should be between xPoins[0] and xPoinst[size-1]. If the value lies outside this range, extrapolation is used."
"\n"
"Regarding spline types:\n"
"  - B: evaluates a cubic B spline approximant.\n"
"  - Beta: evaluates a cubic beta spline approximant. For beta splines parameters 'beta1' and 'beta2' can be supplied. For no bias set beta1 to 1 and for no tension set beta2 to 0. Note that if beta1=1 and beta2=0, the cubic beta becomes a cubic B spline. On the other hand if beta1=1 and beta2 is large the beta spline turns into a linear spline.\n"
"  - Quadratic: evaluates a piecewise quadratic spline at a point. Note that size of input must be odd.\n"
"\n"
"References:\n"
"  [1] Spline interpolation - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Spline_interpolation");

void Spline::compute() {
  const double& xInput = _xInput.get();
  Real& yOutput = _yOutput.get();

  switch (_type) {
    case B:
      yOutput=(Real)spline_b_val(_xPoints.size(), &_xPoints[0], &_yPoints[0], xInput);
      return;
    case BETA:
      yOutput=(Real)spline_beta_val(_beta1, _beta2, (int)_xPoints.size(), &_xPoints[0], &_yPoints[0], xInput);
      return;
    case QUADRATIC:
      double y;
      double dy; //first_derivative (not used)
      spline_quadratic_val((int)_xPoints.size(), &_xPoints[0], &_yPoints[0], xInput, &y, &dy);
      yOutput=(Real)y;
      return;
    default: // should never get here
      throw EssentiaException("Spline: unknown spline type");
  }
}


void Spline::configure() {
  string type = parameter("type").toString();
  if (type == "b") _type = B;
  else if (type == "beta") _type = BETA;
  else  _type = QUADRATIC;/*if (type == "quadratic")*/ // check already done in declareParameter

  vector<Real> x = parameter("xPoints").toVectorReal();
  vector<Real> y = parameter("yPoints").toVectorReal();
  if (x.size() != y.size() ) {
    throw EssentiaException("parameter 'xPoints' must have the same size than parameter 'yPoints')");
  }
  int size = x.size();
  for (int i=0; i<size-1; ++i) {
    if (x[i]>=x[i+1]) {
      throw EssentiaException("parameter 'xPoints' must be in ascendant order and cannot contain duplicates)");
    }
  }
  _xPoints.resize(size);
  _yPoints.resize(size);

  if ((size&1)==0 && _type==QUADRATIC) {
      throw EssentiaException("size of input must be odd when spline type is quadratic");
  }
  for (int i=0; i<size; ++i) {
    _xPoints[i] = double(x[i]);
    _yPoints[i] = double(y[i]);
  }
  // only used for type beta:
  _beta1 = (double)parameter("beta1").toReal();
  _beta2 = (double)parameter("beta2").toReal();
}
