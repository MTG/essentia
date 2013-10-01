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

#include "barkbands.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* BarkBands::name = "BarkBands";
const char* BarkBands::description = DOC("This algorithm computes the spectral energy contained in a given number of bands, which correspond to an extrapolation of the Bark band scale [1]: \n"
"[0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0, 15500.0, 20500.0, 27000.0]\n"
"\n"
"For each bark band the power-spectrum (mag-squared) is summed. The first two bands [0,100] and [100,200] have been split in half for better resolution. It was observed that beat detection is better when this is done.\n"
"\n"
"This algorithm uses FrequencyBands and thus inherits its input requirements and exceptions.\n"
"\n"
"References:\n"
"  [1] The Bark Frequency Scale,\n"
"  http://ccrma.stanford.edu/~jos/bbt/Bark_Frequency_Scale.html");


void BarkBands::configure() {
  static Real bandsFreq[] = { 0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 510.0, 630.0, 770.0, 920.0, 1080.0, 1270.0, 1480.0, 1720.0, 2000.0, 2320.0, 2700.0, 3150.0, 3700.0, 4400.0, 5300.0, 6400.0, 7700.0, 9500.0, 12000.0, 15500.0, 20500.0, 27000.0 };

  vector<Real> bands = arrayToVector<Real>(bandsFreq);
  bands.resize(parameter("numberBands").toInt() + 1);

  _freqBands->configure("frequencyBands", bands,
                        "sampleRate", parameter("sampleRate"));
}

void BarkBands::compute() {

  const vector<Real>& spectrum = _spectrumInput.get();
  vector<Real>& bands = _bandsOutput.get();

  _freqBands->input("spectrum").set(spectrum);
  _freqBands->output("bands").set(bands);
  _freqBands->compute();
}
