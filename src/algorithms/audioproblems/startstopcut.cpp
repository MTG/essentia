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
const char* StartStopCut::description = DOC(
    "This algorithm outputs if there is a cut at the beginning or at the end "
    "of the audio by measuring the first and last non-silent frames and "
    "comparing them to the actual beginning and end of the audio. If the first "
    "non-silent frame occurs before a time threshold the beginning cut flag is "
    "activated. The same applies to the stop cut flag.\n"
    "\n"
    "Notes: This algorithm is designed to operate file-wise. Use it in "
    "combination of RealAccumulator on the streaming mode.\n"
    "The encoding/decoding process of lossy formats can introduce some padding "
    "at the beginning/end of the file. E.g. an MP3 file encoded and decoded "
    "with LAME using the default parameters will introduce a delay of 1104 "
    "samples [http://lame.sourceforge.net/tech-FAQ.txt]. In this case, the "
    "maximumStartTime can be increased by 1104 ÷ 44100 × 1000 = 25 ms to "
    "prevent misdetections.\n");

void StartStopCut::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _frameSize = parameter("frameSize").toInt();
  _hopSize = parameter("hopSize").toInt();
  _maximumStartTime = parameter("maximumStartTime").toReal() / 1000.f;
  _maximumStopTime = parameter("maximumStopTime").toReal() / 1000.f;
  _threshold = db2pow(parameter("threshold").toReal());

  if (_frameSize < _hopSize)
    throw(EssentiaException(
        "StartStopCut: hopSize has to be smaller or equal than the input "
        "frame size"));

  _frameCutter->configure(INHERIT("frameSize"), INHERIT("hopSize"),
                          INHERIT("frameSize"), "startFromZero", true);
};

void StartStopCut::compute() {
  const vector<Real>& audio = _audio.get();
  int& startCut = _startCut.get();
  int& stopCut = _stopCut.get();

  Real start, stop;
  uint sFrame, eFrame;

  // looks for the first non-silent frame
  findNonSilentFrame(audio, sFrame);

  
  std::vector<Real> reversedAudio = audio;
  std::reverse(reversedAudio.begin(), reversedAudio.end());

  // looks for the last non-silent frame
  findNonSilentFrame(reversedAudio, eFrame);

  // sets the start/stop flags according to the thresholds
  start = (Real)(_hopSize * sFrame) / _sampleRate;
  stop = (Real)(_hopSize * eFrame) / _sampleRate;

  (start < _maximumStartTime) ? startCut = true : startCut = false;
  (stop < _maximumStopTime) ? stopCut = true : stopCut = false;
}

void StartStopCut::findNonSilentFrame(std::vector<Real> audio, uint &nonSilentFrame) {
  std::vector<Real> frame;
  bool silentFrame;
  uint nFrame = 0;

  _frameCutter->input("signal").set(audio);
  _frameCutter->output("frame").set(frame);

  while (true) {
    _frameCutter->compute();

    // if it was the last one (ie: it was empty), then we're done.
    if (!frame.size()) {
      break;
    }

    silentFrame = instantPower(frame) < _threshold;
    if (!silentFrame) {
      nonSilentFrame = nFrame;
      break;
    }

    nFrame++;
  }

  _frameCutter->reset();
}

}  // namespace standard
}  // namespace essentia
