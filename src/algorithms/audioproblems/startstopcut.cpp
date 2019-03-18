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

#include "startstopcut.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* StartStopCut::name = "StartStopCut";
const char* StartStopCut::category = "Audio Problems";
const char* StartStopCut::description = DOC(
  "This algorithm outputs if there is a cut at the beginning or at the end "
  "of the audio by locating the first and last non-silent frames and "
  "comparing their positions to the actual beginning and end of the audio. "
  "The input audio is considered to be cut at the beginning (or the end) and "
  "the corresponding flag is activated if the first (last) non-silent frame "
  "occurs before (after) the configurable time threshold.\n"
  "\n"
  "Notes: This algorithm is designed to operate on the entire (file) audio. "
  "In the streaming mode, use it in combination with RealAccumulator.\n"
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

  _maximumStartSamples = (uint)(_maximumStartTime * _sampleRate) + _frameSize;  
  _maximumStopSamples = (uint)(_maximumStopTime * _sampleRate) + _frameSize;  

  _frameCutter->configure(INHERIT("frameSize"), INHERIT("hopSize"),
                          INHERIT("frameSize"), "startFromZero", true);
};


void StartStopCut::compute() {
  const vector<Real>& audio = _audio.get();
  int& startCut = _startCut.get();
  int& stopCut = _stopCut.get();

  if (audio.size() < _maximumStartSamples)
    throw(EssentiaException(
        "StartStopCut: current maximumStartTime value requires at least ",
        _maximumStartSamples, " samples, but the input file size is just ",
        audio.size()));

  if (audio.size() < _maximumStartSamples)
    throw(EssentiaException(
        "StartStopCut: current maximumStopTime value requires at least ",
        _maximumStopSamples, " samples, but the input file size is just ",
        audio.size()));

  // Looks for the first non-silent frame.
  findNonSilentFrame(audio, startCut, _maximumStartSamples / _hopSize);
  
  // Gets a reversed version of the last _maximumStopTime ms of audio.
  std::vector<Real> reversedAudio(audio.end() - _maximumStopSamples, audio.end());
  std::reverse(reversedAudio.begin(), reversedAudio.end());

  // Looks for the last non-silent frame.
  findNonSilentFrame(reversedAudio, stopCut, _maximumStopSamples / _hopSize);
}


void StartStopCut::findNonSilentFrame(std::vector<Real> audio,
                                      int& nonSilentFrame, uint lastFrame) {
  std::vector<Real> frame;
  uint nFrame = 0;

  _frameCutter->input("signal").set(audio);
  _frameCutter->output("frame").set(frame);

  while (nFrame < lastFrame) {
    _frameCutter->compute();

    // If it was the last one (ie: it was empty), then we're done.
    if (!frame.size())
      break;

    nonSilentFrame = instantPower(frame) > _threshold;
    if (nonSilentFrame)
      break;

    nFrame++;
  }

  _frameCutter->reset();
}

}  // namespace standard
}  // namespace essentia
