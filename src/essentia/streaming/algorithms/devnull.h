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

#ifndef ESSENTIA_DEVNULL_H
#define ESSENTIA_DEVNULL_H

#include "../streamingalgorithm.h"

namespace essentia {
namespace streaming {

template <typename TokenType>
class DevNull : public Algorithm {
 protected:
  Sink<TokenType> _frames;

 public:
  DevNull() : Algorithm() {
    static ForcedMutex _devnullInitMutex;
    static int _devnullId = 0;

    ForcedMutexLocker lock(_devnullInitMutex);

    int devnullId = _devnullId++;
    std::ostringstream name;
    name << "DevNull<" << nameOfType(typeid(TokenType)) << ">[" << devnullId << "]";
    setName(name.str());
    declareInput(_frames, 1, "data", "the incoming data to discard");
    E_DEBUG(EFactory, "Created " << _name);
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
