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

#include "clipper.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Clipper::name = "Clipper";
const char* Clipper::description = DOC("This algorithm clips the input signal to fit between the range given by the min and max parameters.\n"
"\n"
"References:\n"
"  [1] Clipping - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Clipping_(audio)");

void Clipper::configure() {
  _max = parameter("max").toReal();
  _min = parameter("min").toReal();
}

void Clipper::compute() {
  const std::vector<Real>& input = _input.get();
  std::vector<Real>& output = _output.get();
  int size = input.size();
  output.resize(size);
  for (int i = 0; i < size; ++i) {
    if (input[i] > _max) output[i] = _max;
    else if (input[i] < _min) output[i] = _min;
    else output[i] = input[i];
  }
}
