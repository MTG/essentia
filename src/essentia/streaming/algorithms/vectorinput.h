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

#ifndef ESSENTIA_VECTORINPUT_H
#define ESSENTIA_VECTORINPUT_H

#include "../streamingalgorithm.h"

namespace essentia {
namespace streaming {


template <typename TokenType, int acquireSize = 1>
class VectorInput : public Algorithm {
 protected:
  Source<TokenType> _output;

  const std::vector<TokenType>* _inputVector;
  bool _ownVector;
  int _idx;

 public:

  VectorInput(const std::vector<TokenType>* input=0, bool own = false)
    : _inputVector(input), _ownVector(own) {
    setName("VectorInput");
    declareOutput(_output, acquireSize, "data", "the values read from the vector");
    reset();
  }

  VectorInput(std::vector<TokenType>* input, bool own = false)
    : _inputVector(input), _ownVector(own) {
    setName("VectorInput");
    declareOutput(_output, acquireSize, "data", "the values read from the vector");
    reset();
  }

  template <typename Array>
  VectorInput(const Array& inputArray, bool own = true) {
    setName("VectorInput");
    _inputVector = new std::vector<TokenType>(arrayToVector<TokenType>(inputArray));
    _ownVector = true;
    declareOutput(_output, acquireSize, "data", "the values read from the vector");
    reset();
  }

  // TODO: This constructor takes in an Array2D but it converts it to a
  // vector-vector to work with the existing code. Ideally, we would keep the
  // Array2D (don't forget to turn off _ownVector) and read from it directly.

  VectorInput(const TNT::Array2D<Real>& input) {
    setName("VectorInput");

    // convert TNT array to vector-vector
    std::vector<TokenType>* inputVector = new std::vector<TokenType>();
    inputVector->resize(input.dim1());

    for (int i=0; i<input.dim1(); ++i) {
      (*inputVector)[i].resize(input.dim2());
      for (int j=0; j<input.dim2(); ++j) {
        (*inputVector)[i][j] = input[i][j];
      }
    }

    _inputVector = inputVector;
    _ownVector = true;
    declareOutput(_output, acquireSize, "data", "the values read from the vector");
    reset();
  }

  ~VectorInput() {
    clear();
  }

  void clear() {
    if (_ownVector) delete _inputVector;
    _inputVector = 0;
  }

  /**
   * TODO: Should we make a copy of the vector here or only keep the ref?
   */
  void setVector(const std::vector<TokenType>* input, bool own=false) {
    clear();
    _inputVector = input;
    _ownVector = own;
  }

  void reset() {
    Algorithm::reset();
    _idx = 0;
    _output.setAcquireSize(acquireSize);
    _output.setReleaseSize(acquireSize);
  }

  bool shouldStop() const {
    return _idx >= (int)_inputVector->size();
  }

  AlgorithmStatus process() {
    // no more data available in vector. shouldn't be necessary to check,
    // but it doesn't cost us anything to be sure
    EXEC_DEBUG("process()");
    if (shouldStop()) {
      return PASS;
    }

    // if we're at the end of the vector, just acquire the necessary amount of
    // tokens on the output source
    if (_idx + _output.acquireSize() > (int)_inputVector->size()) {
      int howmuch = (int)_inputVector->size() - _idx;
      _output.setAcquireSize(howmuch);
      _output.setReleaseSize(howmuch);
    }

    EXEC_DEBUG("acquiring " << _output.acquireSize() << " tokens");
    AlgorithmStatus status = acquireData();

    if (status != OK) {
      if (status == NO_OUTPUT) {
        throw EssentiaException("VectorInput: internal error: output buffer full");
      }
      // should never get there, right?
      return NO_INPUT;
    }

    TokenType* dest = (TokenType*)_output.getFirstToken();
    const TokenType* src = &((*_inputVector)[_idx]);
    int howmuch = _output.acquireSize();
    fastcopy(dest, src, howmuch);
    _idx += howmuch;

    releaseData();
    EXEC_DEBUG("released " << _output.releaseSize() << " tokens");

    return OK;
  }

  void declareParameters() {}

};

template <typename T>
void connect(VectorInput<T>& v, SinkBase& sink) {
  // optimization: if the sink we're connected to requires a lot of samples at once,
  // we might as well wait to have them all instead of feeding it them one by one
  int size = sink.acquireSize();
  SourceBase& visource = v.output("data");
  if (visource.acquireSize() < size) {
    visource.setAcquireSize(size);
    visource.setReleaseSize(size);
  }
  connect(v.output("data"), sink);
}

template <typename T>
void operator>>(VectorInput<T>& v, SinkBase& sink) {
  connect(v, sink);
}

// TODO: in order to use this function runGenerator should be able to be called
// with a vector
template <typename T>
void connect(std::vector<T>& v, SinkBase& sink) {
  VectorInput<T>* vectorInput = new VectorInput<T>(&v);

  // optimize acquire/release sizes to seldom sink's sizes
  int size = sink.acquireSize();
  SourceBase& source = vectorInput->output("data");
  if (source.acquireSize() < size) {
    source.setAcquireSize(size);
    source.setReleaseSize(size);
  }

  connect(vectorInput->output("data"), sink);
}

template <typename T>
void operator>>(std::vector<T>& v, SinkBase& sink) {
  connect(v, sink);
}

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_VECTORINPUT_H
