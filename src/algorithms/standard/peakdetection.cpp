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

#include "peakdetection.h"
#include "essentiamath.h"
#include "peak.h"

using namespace essentia;
using namespace standard;
using namespace util; // peak class

const char* PeakDetection::name = "PeakDetection";
const char* PeakDetection::description = DOC("The peak detection algorithm detects local maxima (peaks) in a data array.\n"
"The algorithm finds positive slopes and detects a peak when the slope changes sign and the peak is above the threshold.\n"
"It optionally interpolates using parabolic curve fitting.\n"
"\n"
"Exceptions are thrown if parameter \"minPosition\" is greater than parameter \"maxPosition\", also if the size of the input array is less than 2 elements.\n"
"\n"
"References:\n"
"  [1] Peak Detection,\n"
"  http://ccrma.stanford.edu/~jos/parshl/Peak_Detection_Steps_3.html");

void PeakDetection::configure() {
  _minPos = parameter("minPosition").toReal();
  _maxPos = parameter("maxPosition").toReal();
  _threshold = parameter("threshold").toReal();
  _maxPeaks = parameter("maxPeaks").toInt();
  _range = parameter("range").toReal();
  _interpolate = parameter("interpolate").toBool();
  _orderBy = parameter("orderBy").toLower();

  if (_minPos >= _maxPos) {
    throw EssentiaException("PeakDetection: The minimum position has to be less than the maximum position");
  }

  // blunt test to make sure some compiler which we won't name isn't going berserk...
  std::vector<Peak> v;
  v.resize(1);
  assert(v.size() == 1);
}

void PeakDetection::compute() {

  const std::vector<Real>& array = _array.get();
  std::vector<Real>& peakValue = _values.get();
  std::vector<Real>& peakPosition = _positions.get();

  const int size = (int)array.size();

  if (size < 2) {
    throw EssentiaException("PeakDetection: The size of the array must be at least 2, for the peak detection to work");
  }

  // dividing by array.size()-1 means the last bin is included in the range
  // dividing by array.size() means it is not (like STL's end interator)
  // which makes more sense in general?
  const Real scale = _range / (Real)(size - 1);

  std::vector<Peak> peaks;
  peaks.reserve(size);

  // we want to round up to the next integer instead of simple truncation,
  // otherwise the peak frequency at i can be lower than _minPos
  int i = std::max(0, (int) ceil(_minPos / scale));

  // first check the boundaries:
  if (i+1 < size && array[i] > array[i+1]) {
    if (array[i] > _threshold) {
      peaks.push_back(Peak(i*scale, array[i]));
    }
  }

  while(true) {
    // going down
    while (i+1 < size-1 && array[i] >= array[i+1]) {
      i++;
    }

    // now we're climbing
    while (i+1 < size-1 && array[i] < array[i+1]) {
      i++;
    }

    // not anymore, go through the plateau
    int j = i;
    while (j+1 < size-1 && (array[j] == array[j+1])) {
      j++;
    }

    // end of plateau, do we go up or down?
    if (j+1 < size-1 && array[j+1] < array[j] && array[j] > _threshold) { // going down again
      Real resultBin = 0.0;
      Real resultVal = 0.0;

      if (j != i) { // plateau peak between i and j
        if (_interpolate) {
          resultBin = (i + j) * 0.5;
        }
        else {
          resultBin = i;
        }
        resultVal = array[i];
      }
      else { // interpolate peak at i-1, i and i+1
        if (_interpolate) {
          interpolate(array[j-1], array[j], array[j+1], j, resultVal, resultBin);
        }
        else {
          resultBin = j;
          resultVal = array[j];
        }
      }

      Real resultPos = resultBin * scale;

      if (resultPos > _maxPos)
        break;

      peaks.push_back(Peak(resultPos, resultVal));
    }

    // nothing found, start loop again
    i = j;

    if (i+1 >= size-1) { // check the one just before the last position
      if (i == size-2 && array[i-1] < array[i] &&
          array[i+1] < array[i] &&
          array[i] > _threshold) {
        Real resultBin = 0.0;
        Real resultVal = 0.0;
        if (_interpolate) {
          interpolate(array[i-1], array[i], array[i+1], j, resultVal, resultBin);
        }
        else {
          resultBin = i;
          resultVal = array[i];
        }
        peaks.push_back(Peak(resultBin*scale, resultVal));
      }
      break;
    }
  }

  // check upper boundary here, so peaks are already sorted by position
  float pos = _maxPos/scale;
  if (size-2 <pos && pos <= size-1 && array[size-1] > array[size-2]) {
    if (array[size-1] > _threshold) {
      peaks.push_back(Peak((size-1)*scale, array[size-1]));
    }
  }

  // we only want this many peaks
  int nWantedPeaks = std::min((int)_maxPeaks, (int)peaks.size());

  if (_orderBy == "amplitude") {
    // sort peaks by magnitude, in case of equality,
    // return the one having smaller position
    std::sort(peaks.begin(), peaks.end(),
              ComparePeakMagnitude<std::greater<Real>, std::less<Real> >());
  }
  else if (_orderBy == "position") {
    // they're already sorted by position
  }
  else {
    throw EssentiaException("PeakDetection: Unsupported ordering type: '" + _orderBy + "'");
  }

  peakPosition.resize(nWantedPeaks);
  peakValue.resize(nWantedPeaks);

  for (int k=0; k<nWantedPeaks; k++) {
    peakPosition[k] = peaks[k].position;
    peakValue[k] = peaks[k].magnitude;
  }
}

/**
* http://ccrma.stanford.edu/~jos/parshl/Peak_Detection_Steps_3.html
*
* Estimating the "true" maximum peak (frequency and magnitude) of the detected local maximum
* using a parabolic curve-fitting. The idea is that the main-lobe of spectrum of most analysis
* windows on a dB scale looks like a parabola and therefore the maximum of a parabola fitted
* through a local maximum bin and it's two neighboring bins will give a good approximation
* of the actual frequency and magnitude of a sinusoid in the input signal.
*
* The parabola f(x) = a(x-n)^2 + b(x-n) + c can be completely described using 3 points;
* f(n-1) = A1, f(n) = A2 and f(n+1) = A3, where
* A1 = 20log10(|X(n-1)|), A2 = 20log10(|X(n)|), A3 = 20log10(|X(n+1)|).
*
* Solving these equation yields: a = 1/2*A1 - A2 + 1/2*A3, b = 1/2*A3 - 1/2*A1 and
* c = A2.
*
* As the 3 bins are known to be a maxima, solving d/dx f(x) = 0, yields the fractional bin
* position x of the estimated peak. Substituting delta_x for (x-n) in this equation yields
* the fractional offset in bins from n where the peak's maximum is.
*
* Solving this equation yields: delta_x = 1/2 * (A1 - A3)/(A1 - 2*A2 + A3).
*
* Computing f(n+delta_x) will estimate the peak's magnitude (in dB's):
* f(n+delta_x) = A2 - 1/4*(A1-A3)*delta_x.
*/
void PeakDetection::interpolate(const Real leftVal, const Real middleVal, const Real rightVal, int currentBin, Real& resultVal, Real& resultBin) const {
  Real delta_x = 0.5 * ((leftVal - rightVal) / (leftVal - 2*middleVal + rightVal));
  resultBin = currentBin + delta_x;
  resultVal = middleVal - 0.25 * (leftVal - rightVal) * delta_x;
}
