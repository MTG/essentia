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

#ifndef ESSENTIA_SOURCE_H
#define ESSENTIA_SOURCE_H

#include "sourcebase.h"
#include "multiratebuffer.h"

namespace essentia {
namespace streaming {


// also known as Output-port, OutputDataStream
template<typename TokenType>
class Source : public SourceBase {
  USE_TYPE_INFO(TokenType);

 protected:
  MultiRateBuffer<TokenType>* _buffer;

 public:

  Source(Algorithm* parent = 0);
  Source(const std::string& name);

  ~Source() {
    delete _buffer;
  }

  const void* buffer() const { return _buffer; }
  void* buffer() { return _buffer; }

  const MultiRateBuffer<TokenType>& typedBuffer() const { return *_buffer; }
  MultiRateBuffer<TokenType>& typedBuffer() { return *_buffer; }

  virtual void setBufferType(BufferUsage::BufferUsageType type) {
    _buffer->setBufferType(type);
  }

  virtual BufferInfo bufferInfo() const {
    return _buffer->bufferInfo();
  }

  virtual void setBufferInfo(const BufferInfo& info) {
    _buffer->setBufferInfo(info);
  }

  int totalProduced() const { return _buffer->totalTokensWritten(); }

  ReaderID addReader() {
    return _buffer->addReader();
  }

  void removeReader(ReaderID id) {
    _buffer->removeReader(id);
  }


  std::vector<TokenType>& tokens() { return _buffer->writeView(); }
  TokenType& firstToken() { return _buffer->writeView()[0]; }
  const TokenType& lastTokenProduced() const { return _buffer->lastTokenProduced(); }

  virtual void* getTokens() { return &tokens(); }
  virtual void* getFirstToken() { return &firstToken(); }

  inline void acquire() { StreamConnector::acquire(); }

  virtual bool acquire(int n) {
    return _buffer->acquireForWrite(n);
  }

  inline void release() { StreamConnector::release(); }

  virtual void release(int n) {
    _buffer->releaseForWrite(n);
  }

  virtual int available() const {
    return _buffer->availableForWrite(false);
  }

  virtual void reset() {
    _buffer->reset();
  }

};

/**
 * AbsoluteSource is a special type of Source that keeps the tokens produced so
 * that Sinks connected after they have been produced still can access them.
 */
template<typename TokenType>
class AbsoluteSource : public Source<TokenType> {
 public:
  ReaderID addReader() {
    return this->_buffer->addReader(true);
  }
};



/**
 * Helper function to be able to call lastTokenProduced on an un-typed SourceBase.
 */
template <typename T>
const T& lastTokenProduced(const SourceBase& source) {
  const Source<T>* src = dynamic_cast<const Source<T>*>(&source);
  if (!src)
    throw EssentiaException(source.fullName(), " does not produce ", nameOfType(typeid(T)), " tokens");

  return src->lastTokenProduced();
}

} // namespace essentia
} // namespace streaming


////////////// IMPLEMENTATION goes here
// NB: Implementation needs to go into the header as it is a template class we are defining

#include "phantombuffer.h"


namespace essentia {
namespace streaming {

// We need to have a specific MultiRateBuffer implementation (PhantomBuffer, here)
// before we can define the constructors
template <typename TokenType>
Source<TokenType>::Source(Algorithm* parent) :
  SourceBase(parent),
  _buffer(new PhantomBuffer<TokenType>(this, BufferUsage::forSingleFrames)) {}

template <typename TokenType>
Source<TokenType>::Source(const std::string& name) :
  SourceBase(name),
  _buffer(new PhantomBuffer<TokenType>(this, BufferUsage::forSingleFrames)) {}

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_SOURCE_H
