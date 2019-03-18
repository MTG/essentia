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

#include "gapsdetector.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char *GapsDetector::name = "GapsDetector";
const char *GapsDetector::category = "Audio Problems";
const char *GapsDetector::description = DOC("This algorithm uses energy "
  "and time thresholds to detect gaps in the waveform. A median filter "
  "is used to remove spurious silent samples. The power of a small "
  "audio region before the detected gaps (prepower) is thresholded to "
  "detect intentional pauses as described in [1]. This technique is"
  "extended to the region after the gap.\n"
  "The algorithm was designed for a framewise use and returns the start "
  "and end timestamps related to the first frame processed. Call "
  "configure() or reset() in order to restart the count.\n"
  "\n"
  "References:\n"
  "  [1] Mühlbauer, R. (2010). Automatic Audio Defect Detection.\n");


void GapsDetector::configure() {
  _sampleRate = parameter("sampleRate").toFloat();
  _frameSize = parameter("frameSize").toInt();
  _hopSize = parameter("hopSize").toInt();
  _silenceThreshold = db2amp(parameter("silenceThreshold").toReal());
  _prepowerThreshold =
      pow(db2amp(parameter("prepowerThreshold").toReal()), 2.f);
  _prepowerTime = parameter("prepowerTime").toFloat() / 1000.f;
  _postpowerTime = parameter("postpowerTime").toFloat() / 1000.f;
  _minimumTime = parameter("minimumTime").toFloat() / 1000.f;
  _maximumTime = parameter("maximumTime").toFloat() / 1000.f;

  _medianFilter->configure(INHERIT("kernelSize"));
  _envelope->configure(INHERIT("attackTime"), INHERIT("releaseTime"));

  _prepowerSamples = _prepowerTime * _sampleRate;
  _postpowerSamples = _postpowerTime * _sampleRate;

  _updateSize = std::min(_hopSize, _prepowerSamples);

  if (_frameSize < _hopSize)
    throw(EssentiaException(
      "GapsDetector: hopSize has to be smaller or equal than the input "
      "frame size"));

  _frameCount = 0;
  _lBuffer.assign(_prepowerSamples, 0.f);
  _gaps.clear();
}


void GapsDetector::compute() {
  const std::vector<Real> frame = _frame.get();
  std::vector<Real> &gapsStarts = _gapsStarts.get();
  std::vector<Real> &gapsEnds = _gapsEnds.get();

  // If the frameSize is not properly set we throw an exception instead of  
  // resizing as probably the hop size is mismatching too.
  if (_frameSize != frame.size())
    throw(
        EssentiaException("GapsDetector: frameSize is not matching the actual "
                          "input size. Please make sure frameSize, hopSize and "
                          "sampleRate are properly set so the output units "
                          "make sense."));

  // Fill the right buffer for each gap candidate.
  for (uint i = 0; i < _gaps.size(); i++) {
    if (!_gaps[i].finished && !_gaps[i].active) {
      uint last = std::min(_frameSize, _gaps[i].remaining);
      _gaps[i].remaining -= last;

      _gaps[i].rBuffer.reserve(_gaps[i].rBuffer.size() + last);
      _gaps[i].rBuffer.insert(_gaps[i].rBuffer.end(), frame.begin(),
                              frame.begin() + last);

      if (_gaps[i].remaining <= 0) _gaps[i].finished = true;
    }
  }

  // Finish the gaps when the right buffer is filled.
  std::vector<uint> removeIndexes;
  for (uint i = 0; i < _gaps.size(); i++) {
    if (_gaps[i].finished) {
      removeIndexes.push_back(i);
      Real postPower = instantPower(_gaps[i].rBuffer);
      if (postPower > _prepowerThreshold) {
        if (_minimumTime <= (_gaps[i].end - _gaps[i].start) &&
            (_gaps[i].end - _gaps[i].start) <= _maximumTime) {
          gapsStarts.push_back(_gaps[i].start);
          gapsEnds.push_back(_gaps[i].end);
        }
      }
    }
  }

  std::reverse(removeIndexes.begin(), removeIndexes.end());
  for (uint i = 0; i < removeIndexes.size(); i++)
    _gaps.erase(_gaps.begin() + removeIndexes[i]);

  // Here the current frame processing starts.
  std::vector<Real> x1, x3;

  _envelope->input("signal").set(frame);
  _envelope->output("signal").set(x1);
  _envelope->compute();

  std::vector<Real> x2(x1.size(), 0.f);
  for (uint i = 0; i < _frameSize; i++)
    x1[i] > _silenceThreshold ? x2[i] = 1.f : x2[i] = 0.f;

  _medianFilter->input("array").set(x2);
  _medianFilter->output("filteredArray").set(x3);
  _medianFilter->compute();

  // Round back to binary values.
  for (uint i = 0; i < _frameSize; i++) 
    x3[i] = (int)(x3[i] + 0.5f);

  // We are only processing the non-overlapping part of the frame.
  uint startProc = (int)((_frameSize / 2) - (_hopSize / 2));
  uint endProc = (int)((_frameSize / 2) + (_hopSize / 2));

  // If there is no overlap we skip only the first sample.
  if (startProc == 0)
    startProc = 1;

  // The gaps limits are detected as rising (u) and falling (d)
  // flanks over the energy mask.
  int diff;
  std::vector<uint> uFlanks, dFlanks;
  for (uint i = startProc; i < endProc; i++) {
    diff = x3[i] - x3[i - 1];
    if (diff == 1) uFlanks.push_back(i);
    if (diff == -1) dFlanks.push_back(i);
  }

  // Initialize gap candidates for all the falling flanks.
  if (dFlanks.size() > 0) {
    std::vector<Real> lBuffer(_prepowerSamples, 0.f);
    Real offset = _frameCount * _hopSize;
    for (uint i = 0; i < dFlanks.size(); i++) {
      int pastValues = dFlanks[i] - _prepowerSamples;

      uint k = 0;
      if (pastValues > 0) {
        for (uint j = pastValues; j < dFlanks[i]; j++, k++)
          lBuffer[k] = frame[j];
      } else {
        uint takeFromBuffer = _lBuffer.size() + pastValues;
        for (uint j = takeFromBuffer; j < _lBuffer.size(); j++, k++)
          lBuffer[k] = _lBuffer[j];
        for (uint j = 0; j < dFlanks[i]; j++, k++) 
          lBuffer[k] = frame[j];
      }
      Real prepower = instantPower(lBuffer);
      if (prepower > _prepowerThreshold) {
        _gaps.push_back(gap{
            _postpowerSamples,
            (Real)((offset + dFlanks[i]) / _sampleRate),
            0,
            true,
            false,
            std::vector<Real>(0),
        });
      }
    }
  }

  // Set the gap candidates ends for every rising flank.
  if (uFlanks.size() > 0) {
    int offset = _frameCount * _hopSize;
    for (uint i = 0; i < uFlanks.size(); i++) {
      uint remaining =
          std::max((int)(_postpowerSamples + uFlanks[i] - _frameSize), 0);
      for (uint j = 0; j < _gaps.size(); j++) {
        if (_gaps[j].active) {
          _gaps[j].active = false;
          _gaps[j].end = (Real)((offset + uFlanks[i]) / _sampleRate);
          _gaps[j].remaining = remaining;
          uint last = std::min(_frameSize, uFlanks[i] + _postpowerSamples);
          std::vector<Real> rBuffer(last - uFlanks[i], 0.f);
          uint l = 0;
          _gaps[j].rBuffer.resize(_frameSize - uFlanks[i]);
          for (uint k = uFlanks[i]; k < _frameSize; k++, l++)
            _gaps[j].rBuffer[l] = frame[k];
          break;
        }
      }
    }
  }

  // Update the left buffer (past values).
  std::rotate(_lBuffer.begin(), _lBuffer.begin() + _updateSize, _lBuffer.end());
  for (uint i = 0; i < _updateSize; i++)
    _lBuffer[_prepowerSamples - _updateSize + i] =
        frame[_frameSize - _updateSize + i];

  _frameCount += 1;
}

void GapsDetector::reset() {
  _frameCount = 0;
  _lBuffer.assign(_prepowerSamples, 0.f);
  _gaps.clear();
}
