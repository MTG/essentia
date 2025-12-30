/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

#include "framebuffer.h"
//#include <cmath>
#include <algorithm>

using namespace std;
using namespace essentia;
using namespace standard;


const char* FrameBuffer::name = "FrameBuffer";
const char* FrameBuffer::category = "Standard";
const char* FrameBuffer::description = DOC(
"This algorithm buffers input non-overlapping audio frames into longer overlapping frames with a hop sizes equal to input frame size.\n\n"
"In standard mode, each compute() call updates and outputs the gathered buffer.\n\n"
"Input frames can be of variate length. Input frames longer than the buffer size will be cropped. Empty input frames will raise an exception."
);


void FrameBuffer::configure() {
  _bufferSize = parameter("bufferSize").toInt();
  _zeroPadding = parameter("zeroPadding").toBool();
  _buffer.resize(_bufferSize);
  reset();
}

void FrameBuffer::reset() {
  if (_zeroPadding) {
    std::fill(_buffer.begin(), _buffer.end(), (Real) 0.);
    _bufferUndefined = 0;
  }
  else {
    _bufferUndefined = _bufferSize;
  }
}

void FrameBuffer::compute() {
  const vector<Real>& frame = _frame.get();
  vector<Real>& bufferedFrame = _bufferedFrame.get();

  if (frame.empty()) throw EssentiaException("FrameBuffer: the input frame is empty");

  int shift = (int) frame.size();
  
  if (shift >= _bufferSize) {
    // Overwrite the entire buffer.
    std::copy(frame.end() - _bufferSize, frame.end(), _buffer.begin());
    _bufferUndefined = 0;
    // TODO E_WARNING for the case of shift > _bufferSize (not all input values fit the buffer)
  }
  else {
    std::copy(_buffer.begin() + shift, _buffer.end(), _buffer.begin());
    std::copy(frame.begin(), frame.end(), _buffer.begin() + _bufferSize - shift);
    if (_bufferUndefined) {
        _bufferUndefined -= shift;
        if (_bufferUndefined < 0) {
            _bufferUndefined = 0;
        }
    }
  }

  // output
  if (!_bufferUndefined) {
    bufferedFrame.resize(_bufferSize);
    std::copy(_buffer.begin(), _buffer.end(), bufferedFrame.begin());
  }
  else {
    // Return emtpy frames until a full buffer is available.
    bufferedFrame.clear();
  }
}
