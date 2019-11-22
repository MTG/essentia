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

#include "triangularbands.h"
#include "essentiamath.h"

namespace essentia {
namespace standard {

const char* TriangularBands::name = "TriangularBands";
const char* TriangularBands::category = "Spectral";
const char* TriangularBands::description = DOC("This algorithm computes energy in triangular frequency bands of a spectrum. The arbitrary number of overlapping bands can be specified. For each band the power-spectrum (mag-squared) is summed.\n"
"\n"
"Parameter \"frequencyBands\" must contain at least two frequencies, they all must be positive and must be ordered ascentdantly, otherwise an exception will be thrown. TriangularBands is only defined for spectrum, which size is greater than 1.\n");


void TriangularBands::configure() {
  _bandFrequencies = parameter("frequencyBands").toVectorReal();
  _nBands = int(_bandFrequencies.size() - 2);
  _inputSize = parameter("inputSize").toReal();
  _sampleRate = parameter("sampleRate").toReal();
  _normalize = parameter("normalize").toLower();
  _type = parameter("type").toLower();
  if ( _bandFrequencies.size() < 2 ) {
    throw EssentiaException("TriangularBands: the 'frequencyBands' parameter contains only one element (at least two elements are required)");
  }

  if (_bandFrequencies.front() < 0) {
    throw EssentiaException("TriangularBands: the 'frequencyBands' parameter contains a negative value");
  }
  for (int i=1; i < (int)_bandFrequencies.size(); ++i) {
    if (_bandFrequencies[i-1] >= _bandFrequencies[i]) {
      throw EssentiaException("TriangularBands: the values in the 'frequencyBands' parameter are not in ascending order or there exists a duplicate value");
    }
  }

  _isLog = parameter("log").toBool();
  setWeightingFunctions(parameter("weighting").toString());
  createFilters(_inputSize);
}


void TriangularBands::compute() {
  const vector<Real>& spectrum = _spectrumInput.get();
  vector<Real>& bands = _bandsOutput.get();

  if (spectrum.size() <= 1) {
    throw EssentiaException("TriangularBands: the size of the input spectrum is not greater than one");
  }

  if (_filterCoefficients.empty() || _filterCoefficients[0].size() != spectrum.size()) {
      E_INFO("TriangularBands: input spectrum size (" << spectrum.size() << ") does not correspond to the \"inputSize\" parameter (" << _filterCoefficients[0].size() << "). Recomputing the filter bank.");
    createFilters(spectrum.size());
  }

  Real frequencyScale = (_sampleRate / 2.0) / (spectrum.size() - 1);

  bands.resize(_nBands);
  fill(bands.begin(), bands.end(), (Real) 0.0);

  for (int i=0; i<_nBands; ++i) {

    // Find margins for FFT bins to iterate through
    // (all bins fall inside the triangle and therefore have non-zero weights).
    int jbegin = ceil(_bandFrequencies[i] / frequencyScale);
    int jend = floor(_bandFrequencies[i+2] / frequencyScale);

    for (int j=jbegin; j<=jend; ++j) {

      if (_type == "power") {
        bands[i] += (spectrum[j] * spectrum[j]) * _filterCoefficients[i][j];
      }

      if (_type == "magnitude") {
        bands[i] += (spectrum[j]) * _filterCoefficients[i][j];
      }

    }
    if (_isLog) bands[i] = log2(1 + bands[i]);
  }
  
}

void TriangularBands::createFilters(int spectrumSize) {
  /*
  Calculate the filter coefficients...
  Basically what we're doing here is the following. Every filter is
  a triangle starting at frequency [i] and going to frequency [i+2].
  This way we have overlap for each filter with the next and the previous one.

        /\
  _____/  \_________
      i    i+2

  After that we normalize the filter over the whole range making sure it's
  norm is 1.

  We could use the optimized scheme from HTK/CLAM/Amadeus, but this makes
  it so much harder to understand what's going on. And you can't have more
  than half-band overlaps either (if needed).
  */

  if (spectrumSize < 2) {
    throw EssentiaException("TriangularBands: Filter bank cannot be computed from a spectrum with less than 2 bins");
  }

  _filterCoefficients = vector<vector<Real> >(_nBands, vector<Real>(spectrumSize, 0.0));

  Real frequencyScale = (_sampleRate / 2.0) / (spectrumSize - 1);

  for (int i=0; i<_nBands; ++i) {
    Real fstep1 = (*_weighter)(_bandFrequencies[i+1]) - (*_weighter)(_bandFrequencies[i]);
    Real fstep2 = (*_weighter)(_bandFrequencies[i+2]) - (*_weighter)(_bandFrequencies[i+1]);

    // Find margins for FFT bins to iterate through
    // (all bins fall inside the triangle).
    int jbegin = ceil(_bandFrequencies[i] / frequencyScale);
    int jend = floor(_bandFrequencies[i+2] / frequencyScale);

    if (jend >= spectrumSize) {
      throw EssentiaException("TriangularBands: the 'frequencyBands' parameter contains a value above the Nyquist frequency (", _sampleRate/2, " Hz): ", _bandFrequencies.back());
    }

    Real weight = 0.;
    for (int j=jbegin; j<=jend; ++j) {
      Real binfreq = j*frequencyScale;
      // in the ascending part of the triangle...
      if (binfreq < _bandFrequencies[i+1]) {
        _filterCoefficients[i][j] = ((*_weighter)(binfreq) - (*_weighter)(_bandFrequencies[i])) / fstep1;
      }
      // in the descending part of the triangle...
      else if (binfreq >= _bandFrequencies[i+1]) {
        _filterCoefficients[i][j] = ((*_weighter)(_bandFrequencies[i+2]) - (*_weighter)(binfreq)) / fstep2;
      }
      weight += _filterCoefficients[i][j];
    }

    if (!weight) {
      throw EssentiaException("TriangularBands: the number of spectrum bins is insufficient for the specified number of triangular bands. Use zero padding to increase the number of FFT bins.");
    }

    // Normalize the filter weights.
    if (_normalize == "unit_tri") {
      // Use theoretical triangular area instead of the actual sum of weights.
      // This is similar to how the normalization is implemnented in Librosa.
      weight = (fstep1 + fstep2) / 2.;
    }

    if (_normalize == "unit_sum" || _normalize == "unit_tri") {
      for (int j=jbegin; j<=jend; ++j) {
        _filterCoefficients[i][j] = _filterCoefficients[i][j] / weight;
      }
    }
  }
}

void TriangularBands::setWeightingFunctions(std::string weighting) {

  if (weighting == "linear") {
      _weighter = hz2hz;
  }
  else if (weighting == "slaneyMel") {
    _weighter = hz2mel;
  }
  else if (weighting == "htkMel") {
    _weighter = hz2mel10;
  }
  else{
    throw EssentiaException("TriangularBands: Bad 'weighting' parameter");
  }
}

} // namespace standard
} // namespace essentia

