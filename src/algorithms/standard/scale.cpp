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

#include "scale.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Scale::name = "Scale";
const char* Scale::description = DOC("This algorithm scales the audio by the specified factor, using clipping if required.");

void Scale::configure() {
  _factor = parameter("factor").toReal();
  _clipping = parameter("clipping").toBool();
  _maxValue = parameter("maxAbsValue").toReal();
}

void Scale::compute() {
  const vector<Real>& signal = _signal.get();
  vector<Real>& scaled = _scaled.get();

  scaled.resize(signal.size());
  fastcopy(scaled.begin(), signal.begin(), scaled.size());

  // scales first
  for (int i=0; i<(int)scaled.size(); i++) {
    scaled[i] *= _factor;
  }

  // does clipping, if applies
  if (_clipping) {
    for (int i=0; i<(int)scaled.size(); i++) {
      if (scaled[i] > _maxValue) scaled[i] = _maxValue;
      if (scaled[i] < -_maxValue) scaled[i] = -_maxValue;
    }
  }
}
