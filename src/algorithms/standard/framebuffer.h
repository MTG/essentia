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

#ifndef ESSENTIA_FRAMEBUFFER_H
#define ESSENTIA_FRAMEBUFFER_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class FrameBuffer : public Algorithm {

 private:
  Input<std::vector<Real> > _frame;
  Output<std::vector<Real> > _bufferedFrame;

  std::vector<Real> _buffer;
  int _bufferSize;
  bool _zeroPadding;
  int _bufferUndefined;  // Number of undefined values in the buffer (= buffer size for the empty buffer on reset).

 public:
  FrameBuffer() {
    declareInput(_frame, "frame", "the input audio frame");
    declareOutput(_bufferedFrame, "frame", "the buffered audio frame");
  }

  void declareParameters() {
    declareParameter("bufferSize", "the buffer size", "(0,inf)", 2048);
    declareParameter("zeroPadding", "initialize the buffer with zeros (output zero-padded buffer frames if `true`, otherwise output empty frames until a full buffer is accumulated)", "{true,false}", true);
  }
  void compute();
  void configure();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace standard
} // namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class FrameBuffer : public StreamingAlgorithmWrapper {

 protected:

  Sink<std::vector<Real> > _frame;
  Source<std::vector<Real> > _bufferedFrame;

 public:
  FrameBuffer() {
    declareAlgorithm("FrameBuffer");
    declareInput(_frame, TOKEN,"frame");
    declareOutput(_bufferedFrame, TOKEN, "frame");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_FRAMEBUFFER_H
