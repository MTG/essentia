/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_DEVNULL_H
#define ESSENTIA_DEVNULL_H

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

template <typename TokenType>
class DevNull : public Algorithm {
 protected:
  Sink<TokenType> _frames;

 public:
  DevNull() : Algorithm() {
    static int devnullId = 0;
    std::ostringstream name;
    name << "DevNull[" << devnullId++ << "]";
    setName(name.str());
    declareInput(_frames, 1, "data", "the incoming data to discard");
  }

  void declareParameters() {}

  AlgorithmStatus process() {
    int nframes = std::min(_frames.available(),
                           _frames.buffer().bufferInfo().maxContiguousElements);
    nframes = std::max(nframes, 1); // in case phantomsize == 0

    EXEC_DEBUG("Consuming " << nframes << " tokens");

    if (!_frames.acquire(nframes)) {
      EXEC_DEBUG("Could not consume because not enough input tokens");
      return NO_INPUT;
    }

    // do nothing, only release the tokens we just acquired
    _frames.release(nframes);

    return OK;
  }
};


enum DevNullConnector {
  NOWHERE,
  DEVNULL
};

/**
 * Connect a source (eg: the output of an algorithm) to a DevNull, so the data
 * the source outputs does not block the whole processing.
 */
void connect(SourceBase& source, DevNullConnector devnull);

inline void operator>>(SourceBase& source, DevNullConnector devnull) {
  connect(source, devnull);
}


/**
 * Disconnect a source (eg: the output of an algorithm) from a DevNull.
 */
void disconnect(SourceBase& source, DevNullConnector devnull);

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_DEVNULL_H
