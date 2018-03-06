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

#include "startstopcut.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* StartStopCut::name = "StartStopCut";
const char* StartStopCut::category = "Audio Probelms";
const char* StartStopCut::description =
    DOC("This algorithm outputs if there is a cut on the beginning or in "
        "the end of the audio by measuring the first and last non silent " 
        "frames and comparing them to the actual beginning and end of the "
        "audio. There are thresholds to set the number of allowed beginning "
        "and end  silent frames.\n");

void StartStopCut::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _frameSize = parameter("frameSize").toInt();
  _hopSize = parameter("hopSize").toInt();
  _maximumStartTime = parameter("maximumStartTime").toReal() / 1000.f;
  _maximumStopTime = parameter("maximumStopTime").toReal() / 1000.f;
  _threshold = db2pow(parameter("threshold").toReal());

  _frameCutter->configure(INHERIT("frameSize"), INHERIT("hopSize"),
                          INHERIT("frameSize"), "startFromZero", true);
};

void StartStopCut::compute() {
  const vector<Real>& audio = _audio.get();
  int& startCut = _startCut.get();
  int& stopCut = _stopCut.get();

  bool silentFrame;
  uint sFrame, eFrame, nFrame;
  Real start, stop;
  std::vector<Real> frame;

  _frameCutter->input("signal").set(audio);
  _frameCutter->output("frame").set(frame);

  nFrame = 0;
  while (true) {
    _frameCutter->compute();

    // if it was the last one (ie: it was empty), then we're done.
    if (!frame.size()) {
      break;
    }

    silentFrame = instantPower(frame) < _threshold;
    if (!silentFrame) {
      sFrame = nFrame;
      break;
    }

    nFrame++;
  }

  _frameCutter->reset();
  std::vector<Real> reversedAudio = audio;
  std::reverse(reversedAudio.begin(), reversedAudio.end());

  _frameCutter->input("signal").set(reversedAudio);
  _frameCutter->output("frame").set(frame);

  nFrame = 0;
  while (true) {
    _frameCutter->compute();

    // if it was the last one (ie: it was empty), then we're done.
    if (!frame.size()) {
      break;
    }

    silentFrame = instantPower(frame) < _threshold;
    if (!silentFrame) {
      eFrame = nFrame;
      break;
    }

    nFrame++;
  }

  start = (Real)(_hopSize * sFrame) / _sampleRate;
  stop = (Real)(_hopSize * eFrame) / _sampleRate;

  (start < _maximumStartTime) ? startCut = true : startCut = false;
  (stop < _maximumStopTime) ? stopCut = true : stopCut = false;
}

}  // namespace standard
}  // namespace essentia
