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

#ifndef ESSENTIA_REALTIMEINPUT_H
#define ESSENTIA_REALTIMEINPUT_H

#include "streamingalgorithmcomposite.h"
#include "algorithmfactory.h"
#include "rtaudio/RtAudio.h"
#include "ringbufferinput.h"

namespace essentia {
namespace streaming {

class RealTimeInput : public AlgorithmComposite {
 
 protected:
  // struct InputData {
  //   Real* buffer;
  //   unsigned long bufferBytes;
  //   unsigned long totalFrames;
  //   unsigned long frameCounter;
  //   unsigned int channels;
  //   RingBufferInput* ringBuffer;
  // };

  // Source<Real> _output;
  SourceProxy<Real> _output;

  Algorithm* _ringBufferInput;

  unsigned int _bufferSize;
  Real _sampleRate;
  int _device;
  Real _channelOffset;

  RtAudio *_adc;
  RtAudio::StreamParameters _iParams;
  
  // InputData _data;

 public:
  RealTimeInput();
  ~RealTimeInput();
 
  void declareParameters() {
    // declareParameter("channels", "mono or stereo", "{mono,stereo}", "mono");  only mono
    declareParameter("bufferSize", "the buffer size", "(0,inf)", 512);
    declareParameter("sampleRate", "the sample rate. Must ve available on the source", "(0,inf)", 44100.0);
    declareParameter("device", "optional device to use", "(0,inf)", 0);
    declareParameter("channelOffset", "an optional channel offset on the device", "(0,inf)", 0.0);
  };

  void configure();
  // void reset();
  AlgorithmStatus process();

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_ringBufferInput));
    declareProcessStep(SingleShot(this));
  }

  static int input(void *                      /*outputBuffer*/,
                   void *inputBuffer,
                   unsigned int nBufferFrames,
                   double                      /*streamTime*/,
                   RtAudioStreamStatus         /*status*/,
                   void *data);

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_REALTIMEINPUT_H
