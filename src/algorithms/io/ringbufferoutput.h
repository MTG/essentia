#ifndef ESSENTIA_STREAMING_RINGBUFFEROUTPUT_H
#define ESSENTIA_STREAMING_RINGBUFFEROUTPUT_H

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

class RingBufferOutput : public Algorithm {
 protected:
  Sink<Real> _input;
  class RingBufferImpl* _impl;

 public:
  RingBufferOutput();
  ~RingBufferOutput();

  int get(Real* outputData, int max);

  AlgorithmStatus process();

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

#endif // ESSENTIA_STREAMING_RINGBUFFEROUTPUT_H
