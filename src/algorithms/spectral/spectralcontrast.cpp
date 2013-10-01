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

#include "spectralcontrast.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* SpectralContrast::name = "SpectralContrast";
const char* SpectralContrast::description = DOC("The Spectral Contrast feature is based on the Octave Based Spectral Contrast feature as described in [1]. The version implemented here is a modified version to improve discriminative power and robustness. The modifications are described in [2].\n"
"\n"
"References:\n"
"  [1] D.-N. Jiang, L. Lu, H.-J. Zhang, J.-H. Tao, and L.-H. Cai, \"Music type\n"
"  classification by spectral contrast feature,\" in IEEE International\n"
"  Conference on Multimedia and Expo (ICME’02), 2002, vol. 1, pp. 113–116.\n\n"
"  [2] V. Akkermans, J. Serrà, and P. Herrera, \"Shape-based spectral contrast\n"
"  descriptor,\" in Sound and Music Computing Conference (SMC’09), 2009,\n"
"  pp. 143–148.\n");


void SpectralContrast::configure() {

  _neighbourRatio = parameter("neighbourRatio").toReal();
  Real sampleRate = parameter("sampleRate").toReal();
  _frameSize = parameter("frameSize").toInt();

  Real upperBound = parameter("highFrequencyBound").toReal();
  if (upperBound > parameter("sampleRate").toReal() / 2)
    throw EssentiaException("SpectralContrast: highFrequencyBound cannot be higher than the Nyquist frequency");

  Real lowerBound = parameter("lowFrequencyBound").toReal();
  if (lowerBound >= upperBound)
    throw EssentiaException("SpectralContrast: lowFrequencyBound cannot be higher than highFrequencyBound");

  int numberBands = parameter("numberBands").toInt();

  // get staticDistribution
  Real partToScale = 1.0 - parameter("staticDistribution").toReal();

  Real binWidth = sampleRate / _frameSize;

  int lastBins = 0;
  _startAtBin = 0;

  _numberOfBinsInBands.clear();
  _numberOfBinsInBands.resize(numberBands);
  lastBins = int(lowerBound / binWidth);
  _startAtBin = lastBins;

  // Determine how many bins are in each band to start with.
  // The rest of the bands will be distributed logarithmically.
  int  totalNumberOfBins = int(upperBound / binWidth);
       upperBound        = int(partToScale*totalNumberOfBins) * binWidth;
  int  staticBinsPerBand = int((1-partToScale)*totalNumberOfBins) / numberBands;
  Real ratio             = upperBound / lowerBound;
  Real ratioPerBand      = pow(ratio, Real(1.0/numberBands));
  Real currFreq          = lowerBound;

  for (int i=0; i<numberBands; ++i) {
    currFreq = currFreq*ratioPerBand;
    _numberOfBinsInBands[i] = int(currFreq / binWidth - lastBins+staticBinsPerBand);
    lastBins = int(currFreq / binWidth);
  }
}

void SpectralContrast::compute() {
  vector<Real> spectrum = _spectrum.get(); // I want a copy because I'll be transforming it
  if (int(spectrum.size()) != _frameSize/2 + 1) {
    ostringstream msg;
    msg << "SpectralContrast: the size of the input spectrum should be half the frameSize parameter + 1. Current spectrum size is: " << spectrum.size() << " while frameSize is " << _frameSize;
    throw EssentiaException(msg);
  }
  //substitute minReal for a static value that is the same in all architectures. i.e.: 1e-30
  Real minReal = 1e-30; //numeric_limits<Real>::min();

  // get the outputs
  vector<Real>& sc = _spectralcontrast.get();
  vector<Real>& valleys = _valleys.get();

  sc.clear();
  valleys.clear();

  int specIdx = _startAtBin;

  for (int bandIdx=0;
       bandIdx < int(_numberOfBinsInBands.size()) && specIdx < int(spectrum.size());
       ++bandIdx) {
    // get the mean of the band
    Real bandMean = 0;
    for (int i=0;
         i<_numberOfBinsInBands[bandIdx] && specIdx+i < int(spectrum.size());
         ++i) {
      bandMean += spectrum[specIdx+i];
    }
    if (_numberOfBinsInBands[bandIdx] != 0) bandMean /= _numberOfBinsInBands[bandIdx];
    bandMean += minReal;

    // sort the subband (ascending order)
    sort(spectrum.begin()+specIdx,
         spectrum.begin()+std::min(specIdx+_numberOfBinsInBands[bandIdx], int(spectrum.size())));

    // number of bins to take the mean of
    int neighbourBins = int(_neighbourRatio * _numberOfBinsInBands[bandIdx]);
    if (neighbourBins < 1) neighbourBins = 1;

    // valley (FLT_MIN prevents log(0))
    Real sum = 0;
    for (int i=0; i<neighbourBins && specIdx+i < int(spectrum.size()); ++i) {
      sum += spectrum[specIdx+i];
    }
    Real valley = sum/neighbourBins + minReal;

    // peak
    sum = 0;
    for (int i=_numberOfBinsInBands[bandIdx];
         i > _numberOfBinsInBands[bandIdx]-neighbourBins &&
           specIdx+i-1 < int(spectrum.size()) &&
           i > 0;
         --i) {
      sum += spectrum[specIdx+i-1];
    }
    Real peak = sum/neighbourBins + minReal;

    sc.push_back(-1.0 * ( pow( peak/valley, Real(1.0/log(bandMean)) ) ));
    valleys.push_back(log(valley));

    specIdx += _numberOfBinsInBands[bandIdx];
  }
}
