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

#include "bpf.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* BPF::name = "BPF";
const char* BPF::description = DOC("A break point function linearly interpolates between discrete xy-coordinates to construct a continuous function.\n"
"\n"
"Exceptions are thrown when the size the vectors specified in parameters is not equal and at least they contain two elements. Also if the parameter vector for x-coordinates is not sorted ascendantly. A break point function cannot interpolate outside the range specified in parameter \"xPoints\". In that case an exception is thrown.\n "
"\n"
"References:\n"
"  [1] Linear interpolation - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Linear_interpolation");


void BPF::compute() {
  const Real& xInput = _xInput.get();
  Real& yOutput = _yOutput.get();

  yOutput = bpf(xInput);
}


void BPF::configure() {
  bpf.init( parameter("xPoints").toVectorReal(), parameter("yPoints").toVectorReal());
}
