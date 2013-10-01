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

#include "flux.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Flux::name = "Flux";
const char* Flux::description = DOC("This algorithm calculates the spectral flux of a given spectrum. Flux is defined as the L2-norm [1] or L1-norm [2] of the difference between two consecutive frames of the magnitude spectrum. The frames have to be of the same size in order to yield a meaningful result. The default L2-norm is used more commonly.\n"
"\n"
"An exception is thrown if the size of the input spectrum does not equal the previous input spectrum's size.\n"
"\n"
"References:\n"
"  [1] Tzanetakis, G., Cook, P., \"Multifeature Audio Segmentation for\n"
"  Browsing and Annotation\", Proceedings of the 1999 IEEE Workshop on\n"
"  Applications of Signal Processing to Audio and Acoustics, New Paltz,\n"
"  NY, USA, 1999, W99 1-4\n\n"
"  [2] S. Dixon, \"Onset detection revisited\", in International Conference on\n"
"  Digital Audio Effects (DAFx'06), 2006, vol. 120, pp. 133-137.\n\n"
"  [3] http://en.wikipedia.org/wiki/Spectral_flux\n");

void Flux::configure() {
  _norm = parameter("norm").toLower();
  _halfRectify = parameter("halfRectify").toBool();
}

void Flux::compute() {

  const vector<Real>& spectrum = _spectrum.get();
  Real& flux = _flux.get();

  if (_spectrumMemory.empty()) {
    // I'm assuming this conditions means its our first iteration
    _spectrumMemory.resize(spectrum.size());
  }
  else if (spectrum.size() != _spectrumMemory.size()) {
    throw EssentiaException("Flux: the size of the input spectrum does not equal the previous input spectrum's size");
  }

  flux = 0.0;

  if (_norm == "l2" && _halfRectify == false) {
    for (int i=0; i<int(spectrum.size()); ++i) {
      flux += (spectrum[i] - _spectrumMemory[i]) * (spectrum[i] - _spectrumMemory[i]);
    }
    flux = sqrt(flux);
  }
  else if (_norm == "l1" && _halfRectify == false) {
    for (int i=0; i<int(spectrum.size()); ++i) {
      flux += abs(spectrum[i] - _spectrumMemory[i]);
    }
  }
  else if (_norm == "l2" && _halfRectify == true) {
    for (int i=0; i<int(spectrum.size()); ++i) {
      Real diff = spectrum[i] - _spectrumMemory[i];
      if (diff < 0) continue;
      flux += diff * diff;
    }
    flux = sqrt(flux);
  }
  else if (_norm == "l1" && _halfRectify == true) {
    for (int i=0; i<int(spectrum.size()); ++i) {
      Real diff = spectrum[i] - _spectrumMemory[i];
      if (diff < 0) continue;
      flux += diff;
    }
  }
  _spectrumMemory = spectrum;
}
