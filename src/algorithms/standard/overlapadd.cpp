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

#include "overlapadd.h"
#include <cmath>
#include <algorithm>

using namespace std;

namespace essentia {

void processFrame(vector<Real>& tmpFrame, const vector<Real>& windowedFrame,
                  vector<Real>& output, vector<Real> &frameHistory,
                  const int& _frameSize, const int& _hopSize, const float& normalizationGain) {

  bool _zeroPhase = true;
  int i = 0;
  int signalSize = _frameSize;


  if (_zeroPhase) {
    // first half of the windowed signal is the
    // second half of the signal with windowing!
    for (int j=signalSize/2; j<signalSize; j++) {
      tmpFrame[i++] = windowedFrame[j] ;
    }

    // second half of the signal
    for (int j=0; j<signalSize/2; j++) {
      tmpFrame[i++] = windowedFrame[j] ;
    }
  }
  else {  // TODO: never happens because zeroPhaze is hardcoded to true, remove?
    // windowed signal
    for (int j=0; j<signalSize; j++) {
      tmpFrame[i++] = windowedFrame[j] ;
    }

  }

  // init buffer by shifting last frame.  TODO: optimize
  for (int i=0; i<_frameSize - _hopSize; i++) {
    frameHistory[i] = frameHistory[i+_hopSize];
  }

  // set the rest of window to 0
  for (int i= (_frameSize - _hopSize); i<_frameSize; i++) {
    frameHistory[i] = 0.;
  }

  // overlap-add
  for (int i=0; i<_frameSize; i++) {
    frameHistory[i] += tmpFrame[i];
  }

  // output
  for (int i=0; i< _hopSize; i++) {
    output[i] = normalizationGain * frameHistory[i];
  }

}

namespace standard {

const char* OverlapAdd::name = "OverlapAdd";
const char* OverlapAdd::category = "Standard";
const char* OverlapAdd::description = DOC("This algorithm returns the output of an overlap-add process for a sequence of frames of an audio signal. It considers that the input audio frames are windowed audio signals. Giving the size of the frame and the hop size, overlapping and adding consecutive frames will produce a continuous signal. A normalization gain can be passed as a parameter.\n"
"\n"
"Empty input signals will raise an exception.\n"
"\n"
"References:\n"
"  [1] Overlap–add method - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Overlap-add_method");


void OverlapAdd::configure() {
  _frameSize = parameter("frameSize").toInt();
  _hopSize = parameter("hopSize").toInt();
  _gain =  parameter("gain").toReal();

  _normalizationGain = 0.5 * _hopSize * _gain ;
  _frameHistory.resize(_frameSize);
  _tmpFrame.resize(_frameSize);
}

void OverlapAdd::reset() {
}
  
void OverlapAdd::compute() {

  const vector<Real>& windowedFrame = _windowedFrame.get();
  vector<Real>& output = _output.get();

  //vector<AudioSample>& audio = _audio.get();

  if (windowedFrame.empty()) throw EssentiaException("OverlapAdd: the input frame is empty");

  output.resize(_hopSize);

  processFrame(_tmpFrame, windowedFrame, output, _frameHistory, _frameSize,
               _hopSize, _normalizationGain);

}
} // namespace standard

namespace streaming {

const char* OverlapAdd::name = essentia::standard::OverlapAdd::name;
const char* OverlapAdd::category = essentia::standard::OverlapAdd::category;
const char* OverlapAdd::description = essentia::standard::OverlapAdd::description;


void OverlapAdd::reset() {
  Algorithm::reset();
  _frames.setAcquireSize(1); // single frame
  _frames.setReleaseSize(1);
  _output.setAcquireSize(_hopSize);
  _output.setReleaseSize(_hopSize);
}


void OverlapAdd::configure() {
  _frameSize = parameter("frameSize").toInt();
  _hopSize = parameter("hopSize").toInt();

  _normalizationGain =  0.5 * _hopSize * parameter("gain").toReal();
  _frameHistory.resize(_frameSize);
  _tmpFrame.resize(_frameSize);
  reset();
}


AlgorithmStatus OverlapAdd::process() {
  EXEC_DEBUG("process()");
  AlgorithmStatus status = acquireData();
  EXEC_DEBUG("data acquired");

  if (status != OK) {
    if (!shouldStop()) return status;

    int available = input("frame").available();
    if (available == 0) {
      return FINISHED;
    }
    // otherwise, there are still some frames
    return CONTINUE;
  }

  const vector<vector<Real> >& frames = _frames.tokens();
  vector<Real>& output = _output.tokens();

  assert(frames.size() == 1 && (int) output.size() == _hopSize);
  const vector<Real> & windowedFrame = frames[0];

  if (windowedFrame.empty()) throw EssentiaException("OverlapAdd: the input frame is empty");

  processFrame(_tmpFrame, windowedFrame, output, _frameHistory, _frameSize,
               _hopSize, _normalizationGain);

  EXEC_DEBUG("releasing");
  releaseData();
  EXEC_DEBUG("released");

  return OK;
}


} // namespace streaming
} // namespace essentia
