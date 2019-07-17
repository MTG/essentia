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

#include "saturationdetector.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* SaturationDetector::name = "SaturationDetector";
const char* SaturationDetector::category = "Audio Problems";
const char* SaturationDetector::description = DOC(
  "this algorithm outputs the staring/ending locations of the saturated "
  "regions in seconds. Saturated regions are found by means of a tripe "
  "criterion:\n"
  "\t 1. samples in a saturated region should have more energy than a "
  "given threshold.\n"
  "\t 2. the difference between the samples in a saturated region should "
  "be smaller than a given threshold.\n"
  "\t 3. the duration of the saturated region should be longer than a "
  "given threshold.\n"
  "\n"
  "note: The algorithm was designed for a framewise use and the returned "
  "timestamps are related to the first frame processed. Use reset() or "
  "configure() to restart the count.");


void SaturationDetector::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _frameSize = parameter("frameSize").toInt();
  _hopSize = parameter("hopSize").toInt();
  _energyThreshold = db2amp(parameter("energyThreshold").toReal());
  _differentialThreshold = parameter("differentialThreshold").toReal();
  _minimumDuration = parameter("minimumDuration").toReal() / 1000.f;

  if (_frameSize < _hopSize)
    throw(EssentiaException(
        "SaturationDetector: hopSize has to be smaller or equal than the input "
        "frame size"));

  _previousStart = 0;
  _idx = 0;


  _startProc = (int)((_frameSize / 2) - (_hopSize / 2));
  _endProc = (int)((_frameSize / 2) + (_hopSize / 2));

  // The algorithm needs at least 2 samples for initialization.
    if (_startProc < 2)
      _startProc = 2;
};


void SaturationDetector::compute() {
  const vector<Real> cFrame = _frame.get();
  vector<Real>& starts = _starts.get();
  vector<Real>& ends = _ends.get();

  vector<Real> frame = cFrame;
  rectify(frame);

  Real start, end, duration;
  
  std::vector<uint> uFlanks, dFlanks;

  Real delta;
  bool currentMask, pastMask;

  delta = abs(frame[_startProc - 1] - frame[_startProc - 2]);
  pastMask = (frame[_startProc - 1] > _energyThreshold) && 
             (delta < _differentialThreshold);
  if ((pastMask) && (!_previousStart))
      uFlanks.push_back(_startProc - 1);

  // Gets the saturated regions starts and ends for the analized part of the
  // frame.
  for (uint i = _startProc; i < _endProc; i++) {
    delta = abs(frame[i] - frame[i-1]);
    
    currentMask = (frame[i] > _energyThreshold) && 
                  (delta < _differentialThreshold);

    if ((currentMask) && (!pastMask))
      uFlanks.push_back(i);
    if ((!currentMask) && (pastMask))
      dFlanks.push_back(i);

    pastMask = currentMask;
  }

  // If the frame starts in the middle of a saturated region it has to be
  // finished first.
  if (_previousStart && (dFlanks.size() > 0)) {
    start = _previousStart;
    end = (float)(_idx * _hopSize + dFlanks[0]) / _sampleRate;
    duration = end - start;

    if (duration > _minimumDuration) {
      starts.push_back(start);
      ends.push_back(end);
    }
    _previousStart = 0;
    dFlanks.erase(dFlanks.begin());
  }

  // If there is an extra rising flank it means that the last saturated region
  // doesn't finish on this frame.
  if ((uFlanks.size() != dFlanks.size()) && (uFlanks.size() > 0)) {
    _previousStart = (float)(_idx * _hopSize + uFlanks.back() ) / _sampleRate;
    uFlanks.pop_back();
  }

  // If the saturation starts before the analyzed part of the first frame just skip it.
  if ((uFlanks.size() != dFlanks.size()) && (_idx == 0))
    dFlanks.pop_back();

  // If this exception is thrown it means that some extra case should have been considered
  // when dessigning the algorithm.
  if (uFlanks.size() != dFlanks.size())
    throw(
        EssentiaException("SaturationDetector: At this point rising and "
                          "falling are expected to have the same length!"));

  // Output the saturated regions.
  for (uint i = 0; i < uFlanks.size(); i++) {
    start = (float)(_idx * _hopSize + uFlanks[i]) / _sampleRate;
    end = (float)(_idx * _hopSize + dFlanks[i]) / _sampleRate;
    duration = end - start;
    if (duration >= _minimumDuration) {
      starts.push_back(start);
      ends.push_back(end);
    }
  }
  _idx += 1;
}


void SaturationDetector::reset() {
  _idx = 0;
  _previousStart = 0;
}

}  // namespace standard
}  // namespace essentia
