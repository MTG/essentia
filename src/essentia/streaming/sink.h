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

#ifndef ESSENTIA_SINK_H
#define ESSENTIA_SINK_H

#include "multiratebuffer.h"
#include "sinkproxy.h"

namespace essentia {
namespace streaming {


template <typename TokenType>
class Source;

// also known as Input-port, InputDataStream
template <typename TokenType>
class Sink : public SinkBase {
  USE_TYPE_INFO(TokenType);

 public:

  Sink(Algorithm* parent = 0, const std::string& name = "unnamed") :
    SinkBase(parent, name) {}

  Sink(const std::string& name) : SinkBase(name) {}

  inline const MultiRateBuffer<TokenType>& buffer() const {
    if (_source) return *static_cast<const MultiRateBuffer<TokenType>*>(_source->buffer());
    else if (_sproxy) return *static_cast<const MultiRateBuffer<TokenType>*>(_sproxy->buffer());
    else
      throw EssentiaException("Sink ", fullName(), " is not currently connected to another Source");
  }

  inline MultiRateBuffer<TokenType>& buffer() {
    if (_source) return *static_cast<MultiRateBuffer<TokenType>*>(_source->buffer());
    else if (_sproxy) return *static_cast<MultiRateBuffer<TokenType>*>(_sproxy->buffer());
    else
      throw EssentiaException("Sink ", fullName(), " is not currently connected to another Source");
  }


  const std::vector<TokenType>& tokens() const { return buffer().readView(_id); }
  const TokenType& firstToken() const { return buffer().readView(_id)[0]; }
  const TokenType& lastTokenProduced() const { return buffer().lastTokenProduced(); }

  virtual const void* getTokens() const { return &tokens(); }
  virtual const void* getFirstToken() const { return &firstToken(); }

  inline void acquire() { StreamConnector::acquire(); }

  virtual bool acquire(int n) {
    if      (_source) return buffer().acquireForRead(_id, n);
    // NOTE: do not call buffer() here because we would have to give an ID which we're not sure is correct
    // this however makes the previous case (if _source) check for _source twice, and is generally less
    // beautiful. We should find a way to have the ID always synchronized to make the code cleaner
    else if (_sproxy) return _sproxy->acquire(n);
    else
      throw EssentiaException("Cannot acquire for sink ", fullName(), ", which has not been connected.");
  }

  inline void release() { StreamConnector::release(); }

  virtual void release(int n) {
    if (_source)      return buffer().releaseForRead(_id, n);
    else if (_sproxy) return _sproxy->release(n);
    else
      throw EssentiaException("Cannot release for sink ", fullName(), ", which has not been connected.");
  }

  virtual int available() const {
    if (_source)      return buffer().availableForRead(_id);
    else if (_sproxy) return _sproxy->available();
    else
      throw EssentiaException("Cannot get number of available tokens for sink ", fullName(),
                              ", which has not been connected.");
  }

  virtual void reset() {}

  TokenType pop() {
    if (!acquire(1))
      throw EssentiaException("No more tokens available to pop in ", fullName());

    TokenType result = *(TokenType*)getFirstToken();
    release(1);

    return result;
  }


};


} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_SINK_H
