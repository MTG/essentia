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

#include "pitchcontoursegmentation.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* PitchContourSegmentation::name = "PitchContourSegmentation";
const char* PitchContourSegmentation::description = DOC("This algorithm converts a pitch sequence estimated from an audio signal into a set of discrete note event. Each note is defined by its onset time, duration and MIDI pitch value, quantized to the equal tempered scale.\n"
"\n"
"Note segmentation is performed based on pitch contour characteristics (island building) and signal RMS. Notes below an adjustable minimum duration are rejected.\n"
"\n"
"References:\n"
"  [1] R. J. McNab et al., \"Signal processing for melody transcription,\" in Proc. \n"
"  Proc. 19th Australasian Computer Science Conf., 1996");


void PitchContourSegmentation::configure() {
  _minDur = parameter("minDur").toReal();
  _tuningFreq = parameter("tuningFreq").toReal();
  reset();
}

void PitchContourSegmentation::reset() {
  
}


void PitchContourSegmentation::compute() {
  const vector<Real>& pitch = _pitch.get();
  const vector<Real>& signal = _signal.get();


     
}
