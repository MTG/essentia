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

#include "mfcc.h"
#include "essentiamath.h" // lin2db

using namespace std;
using namespace essentia;
using namespace standard;

const char* MFCC::name = "MFCC";
const char* MFCC::description = DOC("This algorithm computes the mel-frequency cepstrum coefficients.\n"
"As there is no standard implementation, the MFCC-FB40 is used by default:\n"
"  - filterbank of 40 bands from 0 to 11000Hz\n"
"  - take the log value of the spectrum energy in each mel band\n"
"  - DCT of the 40 bands down to 13 mel coefficients\n"
"There is a paper describing various MFCC implementations [1].\n"
"\n"
"This algorithm depends on the algorithms MelBands and DCT and therefore inherits their parameter restrictions. An exception is thrown if any of these restrictions are not met. The input \"spectrum\" is passed to the MelBands algorithm and thus imposes MelBands' input requirements. Exceptions are inherited by MelBands as well as by DCT.\n"
"\n"
"References:\n"
"  [1] T. Ganchev, N. Fakotakis, and G. Kokkinakis, \"Comparative evaluation\n"
"  of various MFCC implementations on the speaker verification task,\" in\n"
"  International Conference on Speach and Computer (SPECOM’05), 2005, vol. 1,\n"
"  pp. 191–194.\n\n"
"  [2] Mel-frequency cepstrum - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Mel_frequency_cepstral_coefficient");

void MFCC::configure() {
  _melFilter->configure("inputSize", parameter("inputSize"),
                        "sampleRate", parameter("sampleRate"),
                        "numberBands", parameter("numberBands"),
                        "lowFrequencyBound", parameter("lowFrequencyBound"),
                        "highFrequencyBound", parameter("highFrequencyBound"));

  _dct->configure("inputSize", parameter("numberBands"),
                  "outputSize", parameter("numberCoefficients"));
  _logbands.resize(parameter("numberBands").toInt());
}

void MFCC::compute() {

  // get the inputs and outputs
  const vector<Real>& spectrum = _spectrum.get();
  vector<Real>& mfcc = _mfcc.get();
  vector<Real>& bands = _bands.get();

  // filter the spectrum using a mel-scaled filterbank
  _melFilter->input("spectrum").set(spectrum);
  _melFilter->output("bands").set(bands);
  _melFilter->compute();

  // take the dB amplitude of the spectrum
  for (int i=0; i<int(bands.size()); ++i) {
    _logbands[i] = amp2db(bands[i]);
  }

  // compute the DCT of these bands
  _dct->input("array").set(_logbands);
  _dct->output("dct").set(mfcc);
  _dct->compute();
}
