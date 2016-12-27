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

#include "chordsdetectionbeats.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* ChordsDetectionBeats::name = "ChordsDetectionBeats";
const char* ChordsDetectionBeats::category = "Tonal";
const char* ChordsDetectionBeats::description = DOC("This algorithm estimates chords using pitch profile classes on segments between beats. It is similar to ChordsDetection algorithm, but the chords are estimated on audio segments between each pair of consecutive beats.\n"
"\n"
"Quality: experimental (algorithm needs evaluation)\n"
"\n"
"References:\n"
"  [1] E. Gómez, \"Tonal Description of Polyphonic Audio for Music Content\n"
"  Processing,\" INFORMS Journal on Computing, vol. 18, no. 3, pp. 294–304,\n"
"  2006.\n\n"
"  [2] D. Temperley, \"What's key for key? The Krumhansl-Schmuckler\n"
"  key-finding algorithm reconsidered\", Music Perception vol. 17, no. 1,\n"
"  pp. 65-100, 1999.");

void ChordsDetectionBeats::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _hopSize = parameter("hopSize").toInt();
}

void ChordsDetectionBeats::compute() {
  const vector<vector<Real> >& hpcp = _pcp.get();
  vector<string>& chords = _chords.get();
  vector<Real>& strength = _strength.get();
  const vector<Real>& ticks = _ticks.get(); 
  
  string key;
  string scale;
  Real keyStrength;
  Real firstToSecondRelativeStrength;

  if(ticks.size() < 2) { 
    throw EssentiaException("Ticks vector should contain at least 2 elements.");
  } 

  chords.reserve(ticks.size() - 1); 
  strength.reserve(ticks.size() - 1);

  for (int i=0; i < (int)ticks.size()-1; ++i) {

    Real diffTicks = ticks[i+1] - ticks[i];
    int numFramesTick = int((diffTicks * _sampleRate) / _hopSize);
    int frameStart = int((ticks[i] * _sampleRate) / _hopSize);
    int frameEnd = frameStart + numFramesTick-1;

    if (frameEnd > (int)hpcp.size()-1) break;

    vector<Real> hpcpMedian = medianFrames(hpcp, frameStart, frameEnd);
    normalize(hpcpMedian);

    _chordsAlgo->input("pcp").set(hpcpMedian);
    _chordsAlgo->output("key").set(key);
    _chordsAlgo->output("scale").set(scale);
    _chordsAlgo->output("strength").set(keyStrength);
    _chordsAlgo->output("firstToSecondRelativeStrength").set(firstToSecondRelativeStrength);
    _chordsAlgo->compute();

    if (scale == "minor") {
      chords.push_back(key + 'm');
    }
    else {
      chords.push_back(key);
    }

    strength.push_back(keyStrength);
  } 
}

} // namespace standard
} // namespace essentia
