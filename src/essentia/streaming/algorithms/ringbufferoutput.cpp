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

#include "ringbufferoutput.h"
#include "ringbufferimpl.h"
#include "sourcebase.h"
using namespace std;

namespace essentia {
namespace streaming {

const char* RingBufferOutput::name = "RingBufferOutput";
const char* RingBufferOutput::description = DOC("This algorithm fills an output ringbuffer of type Real that can be read from a different thread then.");

RingBufferOutput::RingBufferOutput() : _impl(0)
{
  declareInput(_input, 1024, "signal", "the input signal that should go into the ringbuffer");
}

RingBufferOutput::~RingBufferOutput()
{
	delete _impl;
}

void RingBufferOutput::configure()
{
	delete _impl;
	_impl = new RingBufferImpl(RingBufferImpl::kSpace,parameter("bufferSize").toInt());
}

int RingBufferOutput::get(Real* outputData, int max)
{
	return _impl->get(outputData,max);
}

AlgorithmStatus RingBufferOutput::process() {
  _impl->waitSpace();

  AlgorithmStatus status = acquireData();
  if (status != OK) return status;

  vector<AudioSample>& inputSignal = *((vector<AudioSample>*)input("signal").getTokens());
  AudioSample* inputData = &(inputSignal[0]);
  int inputSize = inputSignal.size();

  int size = _impl->add(inputData, inputSize);
  if (size != inputSize) throw EssentiaException("Not enough space in ringbuffer at output");
  releaseData();

  return OK;
}

void RingBufferOutput::reset() {
  Algorithm::reset();
  _impl->reset();
}

} // namespace streaming
} // namespace essentia
