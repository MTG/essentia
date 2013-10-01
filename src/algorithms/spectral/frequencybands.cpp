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

#include "frequencybands.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* FrequencyBands::name = "FrequencyBands";
const char* FrequencyBands::description = DOC("This algorithm computes the energy of an input spectrum for an arbitrary number of non overlapping frequency bands. For each band the power-spectrum (mag-squared) is summed.\n"
"\n"
"Parameter \"frequencyBands\" must contain at least 2 frequencies, they all must be positive and must be ordered ascentdantly, otherwise an exception will be thrown. FrequencyBands is only defined for spectra, which size is greater than 1.\n"
"\n"
"References:\n"
"  [1] Frequency Range - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Frequency_band\n\n"
"  [2] Band - Handbook For Acoustic Ecology,\n"
"  http://www.sfu.ca/sonic-studio/handbook/Band.html");

void FrequencyBands::configure() {
  _bandFrequencies = parameter("frequencyBands").toVectorReal();
  _sampleRate = parameter("sampleRate").toReal();
  if ( _bandFrequencies.size() < 2 ) {
    throw EssentiaException("FrequencyBands: the 'frequencyBands' parameter contains only one element (i.e. two elements are required to construct a band)");
  }
  for (int i = 1; i < int(_bandFrequencies.size()); ++i) {
    if ( _bandFrequencies[i] < 0 ) {
      throw EssentiaException("FrequencyBands: the 'frequencyBands' parameter contains a negative value");
    }
    if (_bandFrequencies[i-1] >= _bandFrequencies[i] ) {
      throw EssentiaException("FrequencyBands: the values in the 'frequencyBands' parameter are not in ascending order or there exists a duplicate value");
    }
  }
}

void FrequencyBands::compute() {
  const std::vector<Real>& spectrum = _spectrumInput.get();
  std::vector<Real>& bands = _bandsOutput.get();

  if (spectrum.size() <= 1) {
    throw EssentiaException("FrequencyBands: the size of the input spectrum is not greater than one");
  }

  Real frequencyscale = (_sampleRate / 2.0) / (spectrum.size() - 1);
  int nBands = int(_bandFrequencies.size() - 1);

  bands.resize(nBands);
  std::fill(bands.begin(), bands.end(), (Real) 0.0);

  for (int i=0; i<nBands; i++) {
    int startBin = int(_bandFrequencies[i] / frequencyscale + 0.5);
    int endBin = int(_bandFrequencies[i + 1] / frequencyscale + 0.5);

    if (startBin >= int(spectrum.size())) {
      break;
    }

    if (endBin > int(spectrum.size())) {
      endBin = spectrum.size();
    }

    for (int j=startBin; j<endBin; j++) {
      Real magSquared = spectrum[j] * spectrum[j];
      bands[i] += magSquared;
    }
  }

  // decision: don't scale the bands in any way...
  // this way, when summing the energy, we will get consistent *summed* results
  // for different FFT-sizes, (with zero-overlap)
}
