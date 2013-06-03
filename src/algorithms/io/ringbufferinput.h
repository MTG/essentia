#ifndef ESSENTIA_STREAMING_RINGBUFFERINPUT_H
#define ESSENTIA_STREAMING_RINGBUFFERINPUT_H

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

class RingBufferInput : public Algorithm {
 protected:
  Source<Real> _output;
  class RingBufferImpl* _impl;

 public:
  RingBufferInput();
  ~RingBufferInput();

  void add(Real* inputData, int size);

  AlgorithmStatus process();

  void shouldStop(bool stop) {
    E_DEBUG(EExecution, "RBI should stop...");
  }

  void declareParameters() {
    declareParameter("bufferSize", "the size of the ringbuffer", "", 8192);
  }

  void configure();
  void reset();

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_STREAMING_RINGBUFFERINPUT_H
