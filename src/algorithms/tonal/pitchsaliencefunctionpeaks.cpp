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

#include "pitchsaliencefunctionpeaks.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* PitchSalienceFunctionPeaks::name = "PitchSalienceFunctionPeaks";
const char* PitchSalienceFunctionPeaks::version = "1.0";
const char* PitchSalienceFunctionPeaks::description = DOC("This algorithm computes the peaks of a given pitch salience function.\n"
"\n"
"This algorithm is intended to receive its \"salienceFunction\" input from the PitchSalienceFunction algorithm. The peaks are detected using PeakDetection algorithm. The outputs are two arrays of bin numbers and salience values corresponding to the peaks.\n"
"\n"
"References:\n"
"  [1] Salamon, J., & Gómez E. (2012).  Melody Extraction from Polyphonic Music Signals using Pitch Contour Characteristics.\n"
"      IEEE Transactions on Audio, Speech and Language Processing. 20(6), 1759-1770.\n"
);

void PitchSalienceFunctionPeaks::configure() {

  // salience function covers a range of 5 octaves in cent bins

  Real binResolution = parameter("binResolution").toReal();
  Real minFrequency = parameter("minFrequency").toReal();
  Real maxFrequency = parameter("maxFrequency").toReal();
  Real referenceFrequency = parameter("referenceFrequency").toReal();

  Real numberBins = floor(6000.0 / binResolution) - 1;
  Real binsInOctave = 1200.0 / binResolution;
  Real minBin = max(0.0, floor(binsInOctave * log2(minFrequency/referenceFrequency) + 0.5));
  Real maxBin = max(0.0, floor(binsInOctave * log2(maxFrequency/referenceFrequency) + 0.5));
  maxBin = min(numberBins, maxBin);


  // configure algorithms
  _peakDetection->configure("interpolate", false);
  _peakDetection->configure("range", numberBins);
  _peakDetection->configure("maxPosition", maxBin);
  _peakDetection->configure("minPosition", minBin);
  _peakDetection->configure("maxPeaks", 100);
  _peakDetection->configure("orderBy", "amplitude");
}

void PitchSalienceFunctionPeaks::compute() {
  const vector<Real>& salienceFunction = _salienceFunction.get();

  vector <Real>& salienceBins = _salienceBins.get();
  vector <Real>& salienceValues = _salienceValues.get();

  // find salience function peaks
  _peakDetection->input("array").set(salienceFunction);
  _peakDetection->output("positions").set(salienceBins);
  _peakDetection->output("amplitudes").set(salienceValues);
  _peakDetection->compute();
}

