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



#include "framecutter.h"
#include "essentiamath.h" // for isSilent

using namespace std;

namespace essentia {
namespace standard {

const char* FrameCutter::name = "FrameCutter";
const char* FrameCutter::description = DOC("Given an input buffer this algorithm will return a "
"frame (slice) of constant size every time it is called, and then jump a constant amount of "
"samples in the future.\n"
"When no more frames can be extracted from the input buffer, it will return empty frames\n"
"If any frame could not be complete, because we start before the beginning of the input buffer or "
"go past its end, the output frame will be zero-padded.\n"
"\n"
"The rationale for deciding which is the last frame is the following: we should return "
"as many frames as needed to consume all the information contained in the buffer, but no more.\n"
"This translates into 2 different conditions, depending on whether the algorithm has been "
"configured with startFromZero = true or startFromZero = false:\n"
"  - startFromZero = true: a frame is the last one, whenever we are at or beyond the end of the stream. The last frame will be zero-padded if its size is less than \"frameSize\"\n"
"  - startFromZero = false: a frame is the last one if and only if the center of that frame is at or beyond the end of the stream\n"
"then it is the last one\n"
"In both cases, if the start of a frame is past the end of the buffer, we don't return it and "
"stop processing, meaning that the previous frame that we returned was the last one.");


void FrameCutter::configure() {
  _frameSize = parameter("frameSize").toInt();
  _hopSize = parameter("hopSize").toInt();
  _startFromZero = parameter("startFromZero").toBool();
  _lastFrameToEndOfFile = parameter("lastFrameToEndOfFile").toBool();

  Real ratio = parameter("validFrameThresholdRatio").toReal();
  if (ratio > 0.5 && !_startFromZero) {
    throw EssentiaException("FrameCutter: validFrameThresholdRatio cannot be "
                            "larger than 0.5 if startFromZero is false (this "
                            "is to prevent loss of the first frame which would "
                            "be only half a valid frame since the first frame "
                            "is centered on the beginning of the audio)");
  }
  _validFrameThreshold = (int)round(ratio*_frameSize);

  reset();
}

void FrameCutter::reset() {
  _lastFrame = false;

  if (_startFromZero) {
    _startIndex = 0;
  }
  else {
    _startIndex = -(_frameSize+1)/2; // +1 so that odd frameSize start before
  }
}


void FrameCutter::compute() {

  const vector<Real>& buffer = _buffer.get();
  vector<Real>& frame = _frame.get();

  // if we're already lastFrame or the input stream is empty, don't return any frame
  if (_lastFrame || buffer.empty()) {
    frame.clear();
    return;
  }

  // if we're past the end of stream, don't return anything either
  if (_startIndex >= (int)buffer.size()) {
    frame.clear();
    return;
  }


  frame.resize(_frameSize);
  int idxInFrame = 0;

  // if we're before the beginning of the buffer, fill the frame with 0
  if (_startIndex < 0) {
    int howmuch = min(-_startIndex, _frameSize);
    for (; idxInFrame<howmuch; idxInFrame++) {
      frame[idxInFrame] = (Real)0.0;
    }
  }

  // now, just copy from the buffer to the frame
  int howmuch = min(_frameSize, (int)buffer.size() - _startIndex) - idxInFrame;

  fastcopy(&frame[0]+idxInFrame, &buffer[0]+_startIndex+idxInFrame, howmuch);
  idxInFrame += howmuch;

  // check if the idxInFrame is below the threshold (this would only happen
  // for the last frame in the stream)
  if (idxInFrame < _validFrameThreshold) {
    frame.clear();
    _lastFrame = true;
    return;
  }

  if (_startIndex + idxInFrame >= (int)buffer.size() &&
      _startFromZero && !_lastFrameToEndOfFile) _lastFrame = true;

  if (idxInFrame < _frameSize) {
    if (_startFromZero) {
      if (_lastFrameToEndOfFile) {
        if (_startIndex >= (int)buffer.size()) _lastFrame = true;
      }
      // if we're zero-padding with startFromZero=true, it means we're filling
      // in the last frame, so we'll have to stop after this one
      else _lastFrame = true;
    }
    else {
      // if we're zero-padding and the center of the frame is past the end of the
      // stream, then this is the last frame and we need to stop after this one
      if (_startIndex + _frameSize/2 >= (int)buffer.size()) {
        _lastFrame = true;
      }
    }
    // fill in the frame with 0 until the end of the buffer
    for (; idxInFrame < _frameSize; idxInFrame++) {
      frame[idxInFrame] = (Real)0.0;
    }
  }

  // advance frame position
  _startIndex += _hopSize;
}

} // namespace standard
} // namespace essentia


namespace essentia {
namespace streaming {


const char* FrameCutter::name = standard::FrameCutter::name;
const char* FrameCutter::description = DOC("This algorithm slices the input stream into frames of constant size, which are separated by a constant amount of samples, and outputs them as single tokens in the output stream. If any frame could not be complete because we reached the end of the input stream, the output frame will be zero-padded (if it needs to be output).\n"
"\n"
"The rationale for deciding which is the last frame is the following: we should return as many frames as needed to consume all the information contained in the stream, but no more. This translates into 2 different conditions, depending on whether the algorithm has been configured with startFromZero = true or startFromZero = false:\n"
"  - startFromZero = true: a frame is the last one, whenever we are at or beyond the end of the stream. The last frame will be zero-padded if it's size is less than \"frameSize\"\n"
"  - startFromZero = false: a frame is the last one if and only if the center of that frame is at or beyond the end of the stream\n"
"In both cases, if the start of a frame is past the end of the stream, we don't return it and stop processing, meaning that the previous frame that we returned was the last one.\n");


void FrameCutter::reset() {
  Algorithm::reset();
  //_reschedule = false;
  _streamIndex = 0;
  if (_startFromZero) _startIndex = 0;
  else                _startIndex = -(_frameSize+1)/2;

  _audio.setAcquireSize(_frameSize);
  _audio.setReleaseSize(_hopSize);
  _frames.setAcquireSize(1);
  _frames.setReleaseSize(1);
}

FrameCutter::SilenceType FrameCutter::typeFromString(const std::string& name) const {
  if (name == "keep") return KEEP;
  if (name == "drop") return DROP;
  return ADD_NOISE;
}

void FrameCutter::configure() {
  _frameSize = parameter("frameSize").toInt();
  _hopSize = parameter("hopSize").toInt();
  _silentFrames = typeFromString(parameter("silentFrames").toString());
  _lastFrameToEndOfFile = parameter("lastFrameToEndOfFile").toBool();

  _startFromZero = parameter("startFromZero").toBool();
  if (_startFromZero) {
    _startIndex = 0;
  }
  else {
    _startIndex = -(_frameSize+1)/2;
  }

  Real ratio = parameter("validFrameThresholdRatio").toReal();
  if (ratio > 0.5 && !_startFromZero) {
    throw EssentiaException("FrameCutter: validFrameThresholdRatio cannot be "
                            "larger than 0.5 if startFromZero is false (this "
                            "is to prevent loss of the first frame which would "
                            "be only half a valid frame since the first frame "
                            "is centered on the beginning of the audio)");
  }
  _validFrameThreshold = (int)round(ratio*_frameSize);
  // Adding noise to avoid divisions by zero (in case the user chooses to do so
  // by setting the silentFrames parameter to ADD_NOISE).  The level of such noise
  // is chosen to be -100dB because it will still be detected as a silent frame
  // by essentia::isSilent() and is unhearable by humans
  _noiseAdder->configure("fixSeed", false, "level", -100);
  reset();
}

/*
  FrameCutter algo (pseudocode):

  bool lastFrame = false

  do {
    - we need to know whether we have to zero-pad on the left: ie, _startIndex < 0

    - then, we need to see how many more tokens we need to acquire from the stream to fill a frame: howmuch = _frameSize - _startIndex
      - if there are not enough tokens in the stream (howmuch < available):
        -> if shouldStop = false, just return producedData and wait for more to come
        -> if shouldStop = True, we need to acquire what's left: howmuch = available (here we need to pay attention whether _startFromZero = true or false)
            -> if _startFromZero = true: lastFrame = True
            -> if _startFromZero = false and _startIndex + frameSize/2 > _streamIndex + available: lastFrame = true (if center of frame is past the end of stream)

    doAcquire( howmuch )

    acquireFrame(1)

    frame[0:zp] = 0
    frame[zp:zp+howmuch] = memcpy
    frame[zp+homwuch:frameSize] = 0

    release(frame)
    produceData = true

    if (lastFrame) return true;

  } while(true)


 */

AlgorithmStatus FrameCutter::process() {
  bool lastFrame = false;

  EXEC_DEBUG("process()");

  // if _streamIndex < _startIndex, we need to advance into the stream until we
  // arrive at _startIndex
  if (_streamIndex < _startIndex) {
    // to make sure we can skip that many, use frameSize (buffer has been resized
    // to be able to accomodate at least that many sample before starting processing)
    int skipSize = _frameSize;
    int howmuch = min(_startIndex - _streamIndex, skipSize);
    _audio.setAcquireSize(howmuch);
    _audio.setReleaseSize(howmuch);
    _frames.setAcquireSize(0);
    _frames.setReleaseSize(0);

    if (acquireData() != OK) return NO_INPUT;

    releaseData();
    _streamIndex += howmuch;

    return OK;
  }

  // need to know whether we have to zero-pad on the left: ie, _startIndex < 0
  int zeropadSize = 0;
  int acquireSize = _frameSize;
  int releaseSize = min(_hopSize, _frameSize); // in case hopsize > framesize
  int available = _audio.available();

  // we need this check anyway because we might be at the very end of the stream and try to acquire 0
  // for our last frame, which will unfortunately work, so just get rid of this case right now
  if (available == 0) return NO_INPUT;

  if (_startIndex < 0) {
    // left zero-padding and only acquire  as much as _frameSize + startIndex tokens and should release zero
    acquireSize = _frameSize + _startIndex;
    releaseSize = 0;
    zeropadSize = -_startIndex;
  }

  // if there are not enough tokens in the stream (howmuch < available):
  if (acquireSize >= available) { // has to be >= in case the size of the audio fits exactly with frameSize & hopSize
    if (!shouldStop()) return NO_INPUT; // not end of stream -> return and wait for more data to come

    acquireSize = available; // need to acquire what's left
    releaseSize = _startIndex >= 0 ? min(available, _hopSize) : 0; // cannot release more tokens than there are available
    if (_startFromZero) {
      if (_lastFrameToEndOfFile) {
        if (_startIndex >= _streamIndex+available) lastFrame = true;
      }
      else lastFrame = true;
    }
    else {
      if (_startIndex + _frameSize/2 >= _streamIndex + available) // center of frame >= end of stream
        lastFrame = true;
    }
  }

  _frames.setAcquireSize(1);
  _frames.setReleaseSize(1);
  _audio.setAcquireSize(acquireSize);
  _audio.setReleaseSize(releaseSize);

  /*
  EXEC_DEBUG("zeropadSize: " << zeropadSize
             << "\tacquireSize: " << acquireSize
             << "\treleaseSize: " << releaseSize
             << "\tavailable: " << available
             << "\tlast frame: " << lastFrame
             << "\tstartIndex: " << _startIndex
             << "\tstreamIndex: " << _streamIndex);
  */

  AlgorithmStatus status = acquireData();
  EXEC_DEBUG("data acquired (audio: " << acquireSize << " - frames: 1)");

  if (status != OK) {
    if (status == NO_INPUT) return NO_INPUT;
    if (status == NO_OUTPUT) return NO_OUTPUT;
    throw EssentiaException("FrameCutter: something weird happened.");
  }

  // some semantic description to not get mixed up between the 2 meanings
  // of a vector<Real> (which acts both as a stream of Real tokens at the
  // input and as a single vector<Real> token at the output)
  typedef vector<AudioSample> Frame;

  // get the audio input and copy it as a frame to the output
  const vector<AudioSample>& audio = _audio.tokens();
  Frame& frame = _frames.firstToken();


  frame.resize(_frameSize);

  // left zero-padding of the frame
  int idxInFrame = 0;
  for (; idxInFrame < zeropadSize; idxInFrame++) {
    frame[idxInFrame] = (Real)0.0;
  }

  fastcopy(frame.begin()+idxInFrame, audio.begin(), acquireSize);
  idxInFrame += acquireSize;

  // check if the idxInFrame is below the threshold (this would only happen
  // for the last frame in the stream) and if so, don't produce data
  if (idxInFrame < _validFrameThreshold) {
    E_INFO("FrameCutter: dropping incomplete frame");

    // release inputs (advance to next frame), but not the output frame (we didn't produce anything)
    _audio.release(_audio.releaseSize());
    return NO_INPUT;
  }

  // right zero-padding on the last frame
  for (; idxInFrame < _frameSize; idxInFrame++) {
    frame[idxInFrame] = (Real)0.0;
  }

  _startIndex += _hopSize;

  if (isSilent(frame)) {
    switch (_silentFrames) {
    case DROP:
      E_INFO("FrameCutter: dropping silent frame");

      // release inputs (advance to next frame), but not the output frame (we didn't produce anything)
      _audio.release(_audio.releaseSize());
      return OK;

    case ADD_NOISE: {
      vector<AudioSample> inputFrame(_frameSize, 0.0);
      fastcopy(&inputFrame[0]+zeropadSize, &frame[0], acquireSize);
      _noiseAdder->input("signal").set(inputFrame);
      _noiseAdder->output("signal").set(frame);
      _noiseAdder->compute();
      break;
    }

    // otherwise, don't do nothing...
    case KEEP:
    default:
      ;
    }
  }

  EXEC_DEBUG("produced frame; releasing");
  releaseData();
  _streamIndex += _audio.releaseSize();

  EXEC_DEBUG("released");

  if (lastFrame) return PASS;

  return OK;
}

} // namespace streaming
} // namespace essentia
