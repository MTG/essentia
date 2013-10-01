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

#include "onsets.h"
#include "essentiamath.h"

using namespace TNT;
using namespace std;
using namespace essentia;
using namespace standard;

const char* Onsets::name = "Onsets";
const char* Onsets::description = DOC("This algorithm computes onset times in seconds from an array of detection functions extracted from an audio file.\n"
"\n"
"The main operations are:\n"
"  - normalizing detection functions,\n"
"  - summing detection functions into a global detection function,\n"
"  - smoothing the global detection function,\n"
"  - thresholding the global detection function for silence,\n"
"  - finding the possible onsets using an adaptative threshold,\n"
"  - cleaning operations on the vector of possible onsets,\n"
"  - onsets time conversion.\n"
"\n"
"Note:\n"
"  - This algorithm has been optimized for a frameRate of 44100.0/512.0.\n"
"  - At least one Detection function must be supplied at input.\n"
"  - The number of weights must match the number of detection functions.\n"
"\n"
"As mentioned above, the \"frameRate\" parameter expects a value of 44100/512 (the default), but will work with other values, although the quality of the results is not guaranteed then. An exception is also thrown if the input \"detections\" matrix is empty. Finally, an exception is thrown if the size of the \"weights\" input does not equal the first dimension of the \"detections\" matrix.\n"
"\n"
"References:\n"
"  [1] P. Brossier, J. P. Bello, and M. D. Plumbley, \"Fast labelling of notes\n"
"  in music signals,” in International Symposium on Music Information\n"
"  Retrieval (ISMIR’04), 2004, pp. 331–336.");


void Onsets::configure() {
  _alpha = parameter("alpha").toReal(); // alpha is the proportion of the mean we include to reject the smaller peaks
  _silenceThreshold = parameter("silenceThreshold").toReal();
  _frameRate = parameter("frameRate").toReal();

  // this algorithm has been found to be very dependant on the framerate. And should
  // be rewritten. Rewriting it may break onset detection for loopmash, hence a
  // warning will be displayed for framerates different than (1024-512)/44100.
  if (_frameRate >= 44100.0/512.0 + 1e-4 || _frameRate <= 44100.0/512.0 - 1e-4) {
    E_WARNING("Onsets: " << _frameRate << " is not supported as frame rate."
              << "\nThis implementation depends on a frameRate of 44100.0/512.0."
              << "\nStill going on, but results might not be as good as expected...");
  }

  // the comment below might be related to the moving average filter, which uses 5
  // samples by default.
  //Real delay = parameter("delay").toReal() / _frameRate; // assuming that 5 is the best coef for 1024, 512 and 44100.

  // buffersize is linearly scaled to match a different framerate. This is used by the
  // moving average filter (smoothing) and the thresholding operation (mean & median
  // calculation). This makes the algorithm too dependant on the frameRate
  //_bufferSize = int(delay * _frameRate);

  // don't know why delay is multiplied by _frameRate right after being divided
  // by it, bypassing this step for float accuracy. --rtoscano
  _bufferSize = parameter("delay").toInt();

  // this will soon be dependant on the parameters
  _movingAverage->configure("size", _bufferSize);
}

void Onsets::compute() {
  const Array2D<Real>& detections = _detections.get();
  const vector<Real>& weights = _weights.get();
  vector<Real>& onsets = _onsets.get();

  if (detections.dim1() == 0) {
    throw EssentiaException("Onsets: Passing empty matrix as input");
  }

  if (detections.dim1() != int(weights.size())) {
    throw EssentiaException("Onsets: The size of detection functions and the size of weights cannot be different");
  }

  // Copying the Array2D to a vector of vector, much more easy to normalize
  vector<vector<Real> > detections_norm( detections.dim1(), vector<Real>(detections.dim2()) );
  for (int i=0; i<detections.dim1(); ++i) {
    for (int j=0; j<detections.dim2(); ++j) {
      detections_norm[i][j] = detections[i][j];
    }
    normalize(detections_norm[i]);
  }

  // Summing the detection functions into a global detection function
  vector<Real> detection(detections_norm[0].size(), Real(0.0));
  for (int j=0; j<int(detections_norm[0].size()); ++j) {
    for (int i=0; i<int(detections_norm.size()); ++i) {
      detection[j] += weights[i] * detections_norm[i][j];
    }
  }

  // Smoothing the global detection function with a moving average filter
  vector<Real> detection_ma(detection.size(), Real(0.0));
  _movingAverage->input("signal").set(detection);
  _movingAverage->output("signal").set(detection_ma);
  _movingAverage->compute();

  // Thresholding the global detection function for silence
  Real cumulWeights = accumulate(weights.begin(), weights.end(), 0.0);
  for (int i=0; i<int(detection_ma.size()); ++i) {
    if (detection_ma[i] < (_silenceThreshold * cumulWeights)) {
      detection_ma[i] = 0.0;
    }
  }

  // Finding the possible onsets with the adaptative threshold
  vector<bool> onsetDetection(detection_ma.size(), false);
  vector<Real> buffer(_bufferSize, Real(0.0));
  int index = 0;

  for (int i=1; i<int(onsetDetection.size()); ++i) {
    // Updating the buffer
    buffer[index] = detection_ma[i];
    index = (i+1) % buffer.size();

    // Adaptative threshold calculation
    Real buffer_median = median(buffer);
    Real buffer_mean = mean(buffer);
    Real onset_threshold = buffer_median + _alpha * buffer_mean;

    // Onset detection decision
    onsetDetection[i] = detection_ma[i] > onset_threshold;
  }

  // Cleaning operations + time conversion:
  //   - if there is an isolated onset, remove it (false positive?)
  //   - if there are more than 1 consecutive onsets, keep the first one
  // NB: might need to be rewritten.  Looks good to me --rtoscano
  bool onset;
  for (int i=0; i<int(onsetDetection.size()); ++i) {
    onset = false;

    if (onsetDetection[i]) {
      onset = true;
      // transform 000011110000 into 000010000000
      if (i > 0 && onsetDetection[i-1]) {
        onset = false;
      }
      else if (i != int(onsetDetection.size())-1 &&
               !onsetDetection[i+1]) {
        // transform 00000010000 into 00000000000
        onset = false;
      }
    }

    if (onset) onsets.push_back(i / _frameRate);
  }
}
