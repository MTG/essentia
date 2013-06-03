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
