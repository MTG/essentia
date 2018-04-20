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

#include "besselfunction.h"

using namespace essentia;
using namespace standard;

const char* BesselFunction::name = "BesselFunction";
const char* BesselFunction::category = "Standard";
const char* BesselFunction::description = DOC("");

void BesselFunction::compute() {
  const std::vector<Real>& x = _x.get();
  std::vector<Real>& y = _y.get();

  y.resize(x.size());

  double a;
  a = std::tr1::cyl_bessel_i(0.0, 0.0);
  for(uint i = 0; i < x.size(); i++)
  //  y[i] = std::cyl_bessel_i<double, double>((double)_v, (double)x[i]);
   y[i] = 1.0;
  }

void BesselFunction::configure() {
  _v = parameter("v").toInt();
}
