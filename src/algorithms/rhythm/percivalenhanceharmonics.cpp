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

#include "percivalenhanceharmonics.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;

namespace essentia {
namespace standard {

const char* PercivalEnhanceHarmonics::name = "PercivalEnhanceHarmonics";
const char* PercivalEnhanceHarmonics::category = "Rhythm";
const char* PercivalEnhanceHarmonics::description = DOC("This algorithm implements the 'Enhance Harmonics' step as described in [1]."
"Given an input autocorrelation signal, two time-stretched versions of it (by factors of 2 and 4) are added to the original."
"In this way, peaks with an harmonic relation are boosted.\n"
"For more details check the referenced paper."
"\n"
"\n"
"References:\n"
"  [1] Percival, G., & Tzanetakis, G. (2014). Streamlined tempo estimation based on autocorrelation and cross-correlation with pulses.\n"
"  IEEE/ACM Transactions on Audio, Speech, and Language Processing, 22(12), 1765–1776.\n\n");

void PercivalEnhanceHarmonics::configure() {
}

void PercivalEnhanceHarmonics::compute() {
  const vector<Real>& input = _input.get();
  vector<Real>& output = _output.get();
  
  output = input;
  for (int i=0; i<(int)(input.size()/4); ++i) {
    output[i] += output[2*i] + output[4*i];
  }
}

} // namespace standard
} // namespace essentia
