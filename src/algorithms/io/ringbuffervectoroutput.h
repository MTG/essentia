#ifndef ESSENTIA_STREAMING_RINGBUFFERVECTOROUTPUT_H
#define ESSENTIA_STREAMING_RINGBUFFERVECTOROUTPUT_H

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

class RingBufferVectorOutput : public Algorithm {
 protected:
  Sink< std::vector<Real> > _input;
  class RingBufferImpl* _impl;

 public:
  RingBufferVectorOutput();
  ~RingBufferVectorOutput();

  int get(Real* outputData, int max);

  AlgorithmStatus process();

  void declareParameters() {
    declareParameter("bufferSize", "size of the ringbuffer", "", 8192);
  }

  void configure();
  void reset();

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_STREAMING_RINGBUFFEROUTPUT_H
