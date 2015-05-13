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

#include "overlapadd.h"
#include <cmath>
#include <algorithm>

using namespace std;

namespace essentia {
namespace standard {

const char* OverlapAdd::name = "OverlapAdd";
const char* OverlapAdd::description = DOC(
"This algorithm returns the output of an overlap-add process of a sequence of input audio signal frames. It considers that the input audio frames are windowed audio signals. Giving the size of the frame and the hop size, overlapping and adding consecutive frames with produce a continuous signal. \n"
".\n"
"\n"
"Empty input signals will raise an exception.\n"
"\n"
"References:\n"
"  [1] Overlap-Add - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Overlap-Add\n\n"
);


void OverlapAdd::configure() {
  _frameSize = parameter("frameSize").toInt();
  _hopSize = parameter("hopSize").toInt();

  _frameHistory.resize(_frameSize);

_frameCounter = 0;
}

void OverlapAdd::compute() {

  const vector<Real>& windowedFrame = _windowedFrame.get();
  vector<Real>& output = _output.get();

  //vector<AudioSample>& audio = _audio.get();

  if (windowedFrame.empty()) throw EssentiaException("OverlapAdd: the input frame is empty");

  output.resize(_hopSize);

  vector<Real> tmpFrame; // forzerophase
  tmpFrame.resize(_frameSize);  // TODO: allocate memory for tmpFrame in configure to avoid reallocation on every run

 	int M = _frameSize;
  int hM1 = int(floor((M+1)/2.f));



 // check zero-phase

 ////////////////////////////////////
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
  else {
    // windowed signal
    for (int j=0; j<signalSize; j++) {
      tmpFrame[i++] = windowedFrame[j] ;
    }

  }
 ///////////////////////////////////////



  // init buffer by shifting last frame.  TODO: optimize
  for (int i=0; i<_frameSize - _hopSize; i++) {
    _frameHistory[i] = _frameHistory[i+_hopSize];
  }
  // set the rest of window to 0
    for (int i= (_frameSize - _hopSize); i<_frameSize; i++) {
    _frameHistory[i] = 0.f;
  }

  // overlap-add
  for (int i=0; i<_frameSize; i++) {
    _frameHistory[i] += tmpFrame[i];
    }

// output
    float normalizationGain = 0.5f * _hopSize / float(_frameSize);
  for (int i=0; i< _hopSize; i++) {
    output[i] = normalizationGain * _frameHistory[i];
    }


}


} // namespace standard
} // namespace essentia


namespace essentia {
namespace streaming {

const char* OverlapAdd::name = "OverlapAdd";
const char* OverlapAdd::description = DOC(
"This algorithm returns the output of an overlap-add process of a sequence of input audio signal frames. It considers that the input audio frames are windowed audio signals. Giving the size of the frame and the hop size, overlapping and adding consecutive frames with produce a continuous signal. \n"
".\n"
"\n"
"Empty input signals will raise an exception.\n"
"\n"
"References:\n"
"  [1] Overlap-Add - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Overlap-Add\n\n"
);


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
  _frameHistory.resize(_frameSize);
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

    //input("frame").setAcquireSize(available);
    //input("frame").setReleaseSize(available);
    //output("signal").setAcquireSize(available * _hopSize);
    //output("signal").setReleaseSize(available * _hopSize);

    return CONTINUE;
  }

  const vector<vector<Real> >& frames = _frames.tokens();
  vector<Real>& output = _output.tokens();

  //assert(output.size() == input.size() * _hopSize);
  assert(frames.size() == 1 && output.size() == _hopSize);
  const vector<Real> & windowedFrame = frames[0];

  // TODO -- a copy-paste of code from standard mode
  // TODO    make an auxilary function with some code to reuse in both streaming and standard?
  //         or reimplement standard via streaming to reduce the code
  // TODO there is segfault somewhere in this code below (may be it is present in standard mode also)

  if (windowedFrame.empty()) throw EssentiaException("OverlapAdd: the input frame is empty");


  vector<Real> tmpFrame; // forzerophase
  tmpFrame.resize(_frameSize);

  int M = _frameSize;
  int hM1 = int(floor((M+1)/2.f));


 // check zero-phase

 ////////////////////////////////////
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
  else {
    // windowed signal
    for (int j=0; j<signalSize; j++) {
      tmpFrame[i++] = windowedFrame[j] ;
    }

  }
 ///////////////////////////////////////


  // init buffer by shifting last frame.  TODO: optimize
  for (int i=0; i<_frameSize - _hopSize; i++) {
    _frameHistory[i] = _frameHistory[i+_hopSize];
  }
  // set the rest of window to 0
    for (int i= (_frameSize - _hopSize); i<_frameSize; i++) {
    _frameHistory[i] = 0.f;
  }

  // overlap-add
  for (int i=0; i<_frameSize; i++) {
    _frameHistory[i] += tmpFrame[i];
    }

// output
    float normalizationGain = 0.5f * _hopSize / float(_frameSize);
  for (int i=0; i< _hopSize; i++) {
    output[i] = normalizationGain * _frameHistory[i];
    }


  EXEC_DEBUG("releasing");
  releaseData();
  EXEC_DEBUG("released");

  return OK;
}


} // namespace streaming
} // namespace essentia
