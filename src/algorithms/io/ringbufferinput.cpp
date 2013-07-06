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

#include "ringbufferinput.h"
#include "ringbufferimpl.h"
#include "sourcebase.h"
#include "atomic.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* RingBufferInput::name = "RingBufferInput";
const char* RingBufferInput::description = DOC(
"This algorithm gets data from an input ringbuffer of type Real that is fed into the essentia streaming mode."
);

RingBufferInput::RingBufferInput():_impl(0)
{
  declareOutput(_output, 1024, "signal", "data source of what's coming from the ringbuffer");
  _output.setBufferType(BufferUsage::forAudioStream);
}

RingBufferInput::~RingBufferInput()
{
	delete _impl;
}

void RingBufferInput::configure()
{
	delete _impl;
	_impl = new RingBufferImpl(RingBufferImpl::kAvailable,parameter("bufferSize").toInt());
}

void RingBufferInput::add(Real* inputData, int size)
{
	//std::cerr << "adding " << size << " to ringbuffer with space " << _impl->_space << std::endl;
	int added = _impl->add(inputData,size);
	if (added < size) throw EssentiaException("Not enough space in ringbuffer at input");
}

AlgorithmStatus RingBufferInput::process() {
  //std::cerr << "ringbufferinput waiting" << std::endl;
  _impl->waitAvailable();
  //std::cerr << "ringbufferinput waiting done" << std::endl;

  AlgorithmStatus status = acquireData();

  if (status != OK) {
    //std::cerr << "leaving the ringbufferinput while loop" << std::endl;
    if (status == NO_OUTPUT) throw EssentiaException("internal error: output buffer full");
    return status;
  }

  vector<AudioSample>& outputSignal = _output.tokens();
  AudioSample* outputData = &(outputSignal[0]);
  int outputSize = outputSignal.size();

  //std::cerr << "ringbufferinput getting" << outputSize << endl;
  int size = _impl->get(outputData, outputSize);
  //std::cerr << "got " << size << " from ringbuffer with space " << _impl->_space << std::endl;

  _output.setReleaseSize(size);
  releaseData();

  assert(size);

  return OK;
}

void RingBufferInput::reset() {
  Algorithm::reset();
  _impl->reset();
}

} // namespace streaming
} // namespace essentia
