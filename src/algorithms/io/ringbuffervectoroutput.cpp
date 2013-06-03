#include "ringbuffervectoroutput.h"
#include "ringbufferimpl.h"
#include "sourcebase.h"
using namespace std;

namespace essentia {
namespace streaming {

const char* RingBufferVectorOutput::name = "RingBufferVectorOutput";
const char* RingBufferVectorOutput::description = DOC(
"This algorithm fills an output ringbuffer of type Real that can be read from a different thread then.\n"
"The format in the ringbuffer is one value that is the number of values that make up one frame, "
"followed by the actual frame data."
);

RingBufferVectorOutput::RingBufferVectorOutput() : _impl(0)
{
  declareInput(_input, 4096, "signal", "TODO");
}

RingBufferVectorOutput::~RingBufferVectorOutput()
{
	delete _impl;
}

void RingBufferVectorOutput::configure()
{
	delete _impl;
	_impl = new RingBufferImpl(RingBufferImpl::kSpace,parameter("bufferSize").toInt());
}

int RingBufferVectorOutput::get(Real* outputData, int max)
{
	return _impl->get(outputData,max);
}

AlgorithmStatus RingBufferVectorOutput::process() {
  _impl->waitSpace();

  AlgorithmStatus status = acquireData();
  if (status != OK) return status;

  const vector<AudioSample>& inputSignal = _input.firstToken();
  const AudioSample* inputData = &(inputSignal[0]);
  int inputSize = inputSignal.size();

  Real sizeAsReal = inputSize;
  int size = _impl->add(&sizeAsReal, 1);
  if (size != 1) throw EssentiaException("Not enough space in ringbuffer at output");
  size = _impl->add(inputData, inputSize);
  if (size != inputSize) throw EssentiaException("Not enough space in ringbuffer at output");
  releaseData();

  return OK;
}

void RingBufferVectorOutput::reset() {
  Algorithm::reset();
  _impl->reset();
}

} // namespace streaming
} // namespace essentia
