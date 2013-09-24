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

#ifndef ESSENTIA_VECTOROUTPUT_H
#define ESSENTIA_VECTOROUTPUT_H

#include "../streamingalgorithm.h"

namespace essentia {
namespace streaming {

/**
 * VectorOutput class that pushes all data coming at its input into a std::vector.
 * Note that you don't need to configure the VectorOutput to an optimized acquireSize,
 * as it will figure out by itself what's the maximum number of tokens it can acquire
 * at once, and this in a smart dynamic way.
 */
template <typename TokenType, typename StorageType = TokenType>
class VectorOutput : public Algorithm {
 protected:
  Sink<TokenType> _data;
  std::vector<TokenType>* _v;

 public:
  VectorOutput(std::vector<TokenType>* v = 0) : Algorithm(), _v(v) {
    setName("VectorOutput");
    declareInput(_data, 1, "data", "the input data");
  }

  ~VectorOutput() {
  }

  void declareParameters() {}

  void setVector(std::vector<TokenType>* v) {
    _v = v;
  }

  AlgorithmStatus process() {
    if (!_v) {
      throw EssentiaException("VectorOutput algorithm has no output vector set...");
    }

    EXEC_DEBUG("process()");

    int ntokens = std::min(_data.available(), _data.buffer().bufferInfo().maxContiguousElements);
    ntokens = std::max(1, ntokens);

    EXEC_DEBUG("acquiring " << ntokens << " tokens");
    if (!_data.acquire(ntokens)) {
      return NO_INPUT;
    }

    // copy tokens in the vector
    int curSize = _v->size();
    _v->resize(curSize + ntokens);

    TokenType* dest = &_v->front() + curSize;
    const TokenType* src = &_data.firstToken();

    fastcopy(dest, src, ntokens);
    _data.release(ntokens);

    return OK;
  }

  void reset() {
    //_acquireSize = acquireSize;
  }
};

template <typename TokenType, typename StorageType>
void connect(SourceBase& source, VectorOutput<TokenType, StorageType>& v) {
  connect(source, v.input("data"));
}

template <typename TokenType, typename StorageType>
void operator>>(SourceBase& source, VectorOutput<TokenType, StorageType>& v) {
  connect(source, v);
}

/**
 * Connect a source (eg: the output of an algorithm) to a vector that will
 * serve as storage.
 */
template <typename T>
void connect(SourceBase& source, std::vector<T>& v) {
  VectorOutput<T> * vectorOutput = new VectorOutput<T>(&v);
  connect(source, vectorOutput->input("data"));
}

template <typename T>
void operator>>(SourceBase& source, std::vector<T>& v) {
  connect(source, v);
}


} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_VECTOROUTPUT_H
