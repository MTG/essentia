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

#include "noiseburstdetector.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;
using namespace std;

const char *NoiseBurstDetector::name = "NoiseBurstDetector";
const char *NoiseBurstDetector::category = "Audio Problems";
const char *NoiseBurstDetector::description = DOC(
  "This algorithm detects noise bursts in the waveform by thresholding  the "
  "peaks of the second derivative. The threshold is computed using an "
  "Exponential Moving Average filter over the RMS of the second derivative "
  "of the input frame.");


void NoiseBurstDetector::configure() {
  _thresholdCoeff = parameter("threshold").toReal();
  _silenceThreshold = db2pow(parameter("silenceThreshold").toReal());
  _alpha = parameter("alpha").toReal();

  _threshold = 1.f;
}


void NoiseBurstDetector::compute() {
  const std::vector<Real> frame = _frame.get();
  std::vector<Real> &indexes = _indexes.get();

  if (instantPower(frame) <_silenceThreshold) {
    return;
  }

  // Get the second derivative of the frame.
  vector<Real> ddFrame = derivative(derivative(frame));

  // Update the threshold using Exponential
  // Moving Average.
  updateEMA( _thresholdCoeff * robustRMS(ddFrame, 2));

  for (size_t i = 0; i < ddFrame.size(); i++) {
    if (ddFrame[i] > _threshold) {
      indexes.push_back(i);
    }
  }
}


// The version of RMS smashes the signal according to
// the median in order to reduce the weight of outliers
// samples in the RMS estimation.
Real NoiseBurstDetector::robustRMS(std::vector<Real> x, Real k) {
    for (size_t i = 0; i < x.size(); i++) {
      x[i] *= x[i];
    }

    // Smash the signal to k times the meadian.
    std::vector<Real> robustX;  
    
    _Clipper->configure("max", median(x) * k);
    _Clipper->input("signal").set(x);
    _Clipper->output("signal").set(robustX);
    _Clipper->compute();

    return sqrt(mean(robustX));
}


void NoiseBurstDetector::updateEMA(Real x) {
    _threshold = _threshold * (1 - _alpha) + x * _alpha;
}
