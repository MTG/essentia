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

#include "frametoreal.h"
#include <cmath>
#include <algorithm>

using namespace std;
using namespace essentia;
using namespace standard;


const char* FrameToReal::name = "FrameToReal";
const char* FrameToReal::category = "Standard";
const char* FrameToReal::description = DOC(
"This algorithm converts a sequence of input audio signal frames into a sequence of audio samples.\n\n"
"Empty input signals will raise an exception."
);


void FrameToReal::configure() {
  _frameSize = parameter("frameSize").toInt();
  _hopSize = parameter("hopSize").toInt();

}

void FrameToReal::compute() {

  const vector<Real>& frames = _frames.get();
  vector<Real>& audio = _audio.get();


  if (frames.empty()) throw EssentiaException("FrameToReal: the input signal is empty");

  // output
  audio.resize(_hopSize);
  for (int i=0; i< _hopSize; i++) {
    audio[i] = frames[i];
  }
}
