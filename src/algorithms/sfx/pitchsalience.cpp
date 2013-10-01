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

#include "pitchsalience.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* PitchSalience::name = "PitchSalience";
const char* PitchSalience::description = DOC("This algorithm computes the pitch salience of a spectrum. The pitch salience is given by the ratio of the highest auto correlation value of the spectrum to the non-shifted auto correlation value. Pitch salience was designed as quick measure of tone sensation. Unpitched sounds (non-musical sound effects) and pure tones have an average pitch salience value close to 0 whereas sounds containing several harmonics in the spectrum tend to have a higher value.\n\n"
"Note that this algorithm may give better results when used with low sampling rates (i.e. 8000) as the information in the bands musically meaningful will have more relevance.\n\n"
"This algorithm uses AutoCorrelation on the input \"spectrum\" and thus inherits its input requirements and exceptions. An exception is thrown at configuration time if \"lowBoundary\" is larger than \"highBoundary\" and/or if \"highBoundary\" is not smaller than half \"sampleRate\". At computation time, an exception is thrown if the input spectrum is empty. Also note that feeding silence to this algorithm will return zero.");

void PitchSalience::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _lowBoundary = parameter("lowBoundary").toReal();
  _highBoundary = parameter("highBoundary").toReal();

  if (_lowBoundary > _highBoundary) {
    throw EssentiaException("PitchSalience: lowBoundary is larger than highBoundary");
  }

  if (_highBoundary >= _sampleRate/2) {
    throw EssentiaException("PitchSalience: highBoundary is not smaller than half sampleRate");
  }
}

void PitchSalience::compute() {

  const vector<Real>& spectrum = _spectrum.get();
  Real& pitchSalience = _pitchSalience.get();

  if (spectrum.empty()) {
    throw EssentiaException("PitchSalience: spectrum is an empty vector");
  }

  vector<Real> acf;
  _autoCorrelation->input("array").set(spectrum);
  _autoCorrelation->output("autoCorrelation").set(acf);
  _autoCorrelation->compute();

  int lowIndex = int((_lowBoundary * spectrum.size()) /
                     (_sampleRate/2));
  int highIndex = int((_highBoundary * spectrum.size()) /
                      (_sampleRate/2));

  Real acfMax = *max_element(acf.begin() + lowIndex, acf.begin() + highIndex);

  if (acf[0]  == 0) pitchSalience = 0.0;
  else pitchSalience = acfMax / acf[0];
}
