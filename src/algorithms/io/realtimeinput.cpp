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

#include "realtimeinput.h"

#define FORMAT RTAUDIO_FLOAT32

using namespace std;

using namespace essentia;
using namespace streaming;

const char* RealTimeInput::name = "RealTimeInput";
const char* RealTimeInput::category = "Input/Output";
const char* RealTimeInput::description = DOC(
"This algorithm gets data from an input ringbuffer of type Real that is fed into the essentia streaming mode."
);


RealTimeInput::RealTimeInput() : AlgorithmComposite() {
    declareOutput(_output, 512, "data", "the values read from the vector");
    _adc = new RtAudio(RtAudio::LINUX_ALSA);
    _ringBufferInput->output("signal") >> _output;
}

RealTimeInput::~RealTimeInput() {
  if (_adc->isStreamOpen()) _adc->closeStream();
  
  delete _adc;
}


void RealTimeInput::configure() {
  _bufferSize = parameter("bufferSize").toInt();
  _sampleRate = parameter("sampleRate").toReal();
  _device = parameter("device").toInt();
  _channelOffset = parameter("channelOffset").toReal();

  if (_adc->getDeviceCount() < 1) {
    throw EssentiaException("No audio devices found!");
  }

  // TODO: only if debug
  _adc->showWarnings(true);

  if (_device == 0) {
    _iParams.deviceId = _adc->getDefaultInputDevice();
  }
  else {
    _iParams.deviceId = _device;
  }

  // This algorithm only support mono input.
  _iParams.nChannels = 1;

  _iParams.firstChannel = _channelOffset;

  // _data.buffer = 0;

  _ringBufferInput = AlgorithmFactory::create("RingBufferInput",
                                              "bufferSize", _bufferSize);

  try {
    _adc->openStream(NULL, 
                    &_iParams,
                    FORMAT,
                    _sampleRate,
                    &_bufferSize,
                    &input,
                    (void *)&_ringBufferInput);
  }
  catch (RtAudioError& e) {
    throw EssentiaException(e.getMessage());
  }
}


AlgorithmStatus RealTimeInput::process() {
  if (!_adc->isStreamRunning()) {
    try {
      _adc->startStream();
    }
    catch ( RtAudioError& e ) {
      throw EssentiaException(e.getMessage());
    }
  }
  if (!shouldStop()) return PASS;

  if (_adc->isStreamOpen()) _adc->closeStream();
  return FINISHED;
}


int RealTimeInput::input(void * /*outputBuffer*/,
                                void *inputBuffer,
                                unsigned int nBufferFrames,
                                double /*streamTime*/,
                                RtAudioStreamStatus /*status*/,
                                void *data){
  
  RingBufferInput *ringBuffer = (RingBufferInput *) data;

  ringBuffer->add((Real *)inputBuffer, nBufferFrames);


  return 0;
}
