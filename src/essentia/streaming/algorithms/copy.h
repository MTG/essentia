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

#ifndef ESSENTIA_STREAMING_COPY_H
#define ESSENTIA_STREAMING_COPY_H

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

template <typename TokenType>
class Copy : public Algorithm {
 protected:
  Sink<TokenType> _framesIn;
  Source<TokenType> _framesOut;

 public:
  Copy() : Algorithm() {
    static ForcedMutex _copyInitMutex;
    static int _copyId = 0;

    ForcedMutexLocker lock(_copyInitMutex);

    int copyId = _copyId++;
    std::ostringstream name;
    name << "Copy<" << nameOfType(typeid(TokenType)) << ">[" << copyId << "]";
    setName(name.str());
    declareInput(_framesIn, 1, "data", "the input data");
    declareOutput(_framesOut, 1, "data", "the output data");
    E_DEBUG(EFactory, "Created " << _name);
  }

  void declareParameters() {}

  AlgorithmStatus process() {
    int nframes = std::min(_framesIn.available(),
                           _framesIn.buffer().bufferInfo().maxContiguousElements);
    nframes = std::max(nframes, 1); // in case phantomsize == 0

    EXEC_DEBUG("Consuming " << nframes << " tokens");

    if (!_framesIn.acquire(nframes)) {
      EXEC_DEBUG("Could not consume because not enough input tokens");
      return NO_INPUT;
    }

    if (!_framesOut.acquire(nframes)) {
      EXEC_DEBUG("Could not consume because not enough output tokens");
      return NO_OUTPUT;
    }

    fastcopy(&_framesOut.firstToken(), &_framesIn.firstToken(), nframes);

    // release the tokens we just copied
    _framesIn.release(nframes);
    _framesOut.release(nframes);

    return OK;
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_STREAMING_COPY_H
