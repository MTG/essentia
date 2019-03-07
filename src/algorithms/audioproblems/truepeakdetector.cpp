/*
 * Copyright (C) 2006-2019  Music Technology Group - Universitat Pompeu Fabra
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

#include "truepeakdetector.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;
using namespace std;

const char* TruePeakDetector::name = "TruePeakDetector";
const char* TruePeakDetector::category = "Audio Problems";
const char* TruePeakDetector::description = DOC(
  "This algorithm implements a “true-peak” level meter for clipping detection. "
  "According to the ITU-R recommendations, “true-peak” values overcoming the full-scale range are "
  "potential sources of “clipping in subsequent processes, such as within particular "
  "D/A converters or during sample-rate conversion”.\n"
  "The ITU-R BS.1770-4[1] (by default) and the ITU-R BS.1770-2[2] signal-flows can be used. "
  "Go to the references for information about the differences.\n"
  "Only the peaks (if any) exceeding the configurable amplitude threshold are returned."
  "\n"
  "Note: the parameters 'blockDC' and 'emphasise' work only when 'version' is set to 2."
  "\n"
  "References:\n"
  "  [1] Series, B. S. (2011). Recommendation  ITU-R  BS.1770-4. Algorithms to measure audio programme "
  "loudness and true-peak audio level,\n"
  "  "
  "https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-4-201510-I!!PDF-E.pdf\n"
  "  [2] Series, B. S. (2011). Recommendation  ITU-R  BS.1770-2. Algorithms to measure audio programme "
  "loudness and true-peak audio level,\n"
  "  "
  "https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-2-201103-S!!PDF-E.pdf\n");


void TruePeakDetector::configure() {
  _inputSampleRate = parameter("sampleRate").toReal();
  _oversamplingFactor = parameter("oversamplingFactor").toReal();
  _outputSampleRate = _inputSampleRate * _oversamplingFactor;
  _quality = parameter("quality").toInt();
  _blockDC = parameter("blockDC").toBool();
  _emphasise = parameter("emphasise").toBool();
  _threshold = db2amp(parameter("threshold").toFloat());
  _version = parameter("version").toInt();

  _resampler->configure("inputSampleRate", _inputSampleRate,
                       "outputSampleRate", _outputSampleRate,
                       "quality", _quality);

  if (_emphasise) {
    // The parameters of the filter are extracted from the recommendation.
    Real poleFrequency = 20e3;  // Hz
    Real zeroFrequncy = 14.1e3; // Hz

    Real rPole = 1 - 4 * poleFrequency / _outputSampleRate;
    Real rZero = 1 - 4 * zeroFrequncy / _outputSampleRate;

    vector<Real> b(2, 0.0);
    b[0] = 1.0;
    b[1] = -rZero;

    vector<Real> a(2, 0.0);
    a[0] = 1.0;
    a[1] = rPole;

    _emphasiser->configure( "numerator", b, "denominator", a);
  }

  if (_blockDC) {
    _dcBlocker->configure("sampleRate", _outputSampleRate);
  }
}


void TruePeakDetector::compute() {
  vector<Real>& output = _output.get();
  vector<Real>& peakLocations = _peakLocations.get();

  std::vector<Real> *processed;

  std::vector<Real> resampled;
  _resampler->input("signal").set(_signal.get());
  _resampler->output("signal").set(resampled);
  _resampler->compute();
  processed = &resampled;

  if (_version == 2) {
    if (_emphasise) {
      std::vector<Real> emphasised;
      _emphasiser->input("signal").set(*processed);
      _emphasiser->output("signal").set(emphasised);
      _emphasiser->compute();
      processed = &emphasised;
    }

    if (_blockDC) {
      std::vector<Real> dcBlocked;
      _dcBlocker->input("signal").set(*processed);
      _dcBlocker->output("signal").set(dcBlocked);
      _dcBlocker->compute();
      for (uint i = 0; i < processed->size(); i++)
        (*processed)[i] = max(abs((*processed)[i]), abs(dcBlocked[i]));
    }
  }

  if ((_version == 4) || (!_blockDC))
      rectify((*processed));

    for (uint i = 0; i < processed->size(); i++)
      if ((*processed)[i] >= _threshold)
        peakLocations.push_back((int) (i / _oversamplingFactor));

    output = *processed;
  }
