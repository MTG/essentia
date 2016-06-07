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

#ifndef ESSENTIA_SOURCEBASE_H
#define ESSENTIA_SOURCEBASE_H

#include "../types.h"
#include "../connector.h"

namespace essentia {
namespace streaming {

class SourceBase;
class SourceProxyBase;
  //template <typename T> class SourceProxy;
class SinkBase;
class Algorithm;

void connect(SourceBase& source, SinkBase& sink);
void disconnect(SourceBase& source, SinkBase& sink);

void attach(SourceBase& innerSource, SourceProxyBase& proxy);
void detach(SourceBase& innerSource, SourceProxyBase& proxy);


/**
 * This is the base class from which Sources should derive. It defines the basic
 * interface a Source should provide, such as the acquire() and release() methods,
 * and a way to get hold of one or more tokens that are waiting to be consumed.
 * It is untyped (in the sense that it doesn't know which type are the tokens),
 * but derives from TypeProxy and as such has functions that can do type-checking
 * with respect to the types of the derived Sink (which is a templated class,
 * the template being the token type). Look at the Source implementation for more
 * information.
 */
class SourceBase : public Connector {
 protected:
  std::vector<SinkBase*> _sinks;

  // we only allow for 1 proxy to be connected at the moment
  // (although multiple ones would be theoretically correct, too)
  SourceProxyBase* _sproxy;

 public:
  // TODO: are those still useful?
  SourceBase(Algorithm* parent = 0, const std::string& name = "unnamed") :
    Connector(parent, name), _sproxy(0) {}

  SourceBase(const std::string& name) :
    Connector(name), _sproxy(0) {}

  ~SourceBase();

  // this function should probably be protected, with friend = SinkBase, Sink
  virtual void* buffer() = 0;

  virtual int totalProduced() const = 0;

  const std::vector<SinkBase*>& sinks() const { return _sinks; }

  // TODO: remove me to avoid people doing stuff like src.sinks().clear()
  std::vector<SinkBase*>& sinks() { return _sinks; }

  // should return a vector<TokenType>*
  virtual void* getTokens() = 0;

  // should return a TokenType*
  virtual void* getFirstToken() = 0;

  bool isProxied() const { return _sproxy != 0; }

  /**
   * Return the list of sinks that are connected through a proxy.
   * Make sure to call isProxied() before.
   */
  const std::vector<SinkBase*>& proxiedSinks() const;

  template <typename TokenType>
  void push(const TokenType& value) {
    try {
      checkType<TokenType>();
      if (!acquire(1))
        throw EssentiaException(fullName(), ": Could not push 1 value, output buffer is full");

      *(TokenType*)getFirstToken() = value;

      release(1);
    }
    catch (EssentiaException& e) {
      throw EssentiaException("While trying to push item into source ", fullName(), ":\n", e.what());
    }
  }

  // function to resize the buffer given the type of tokens we want to convey
  virtual void setBufferType(BufferUsage::BufferUsageType type) = 0;

  virtual BufferInfo bufferInfo() const = 0;
  virtual void setBufferInfo(const BufferInfo& info) = 0;

 protected:
  // made those protected so that only our friend streaming::{dis}connect() functions can access these
  // @todo this function should probably be protected by a mutex (?)
  virtual void connect(SinkBase& sink);
  virtual void disconnect(SinkBase& sink);

  friend void connect(SourceBase& source, SinkBase& sink);
  friend void disconnect(SourceBase& source, SinkBase& sink);


  virtual ReaderID addReader() = 0;
  virtual void removeReader(ReaderID id) = 0;


  friend void attach(SourceBase& innerSource, SourceProxyBase& proxy);
  friend void detach(SourceBase& innerSource, SourceProxyBase& proxy);

  // Note: these can't be called attach because they would be shadowed by SourceProxyBase::attach(SourceBase)
  void attachProxy(SourceProxyBase* sproxy);
  void detachProxy(SourceProxyBase* sproxy);

  friend class SourceProxyBase;
  // TODO: still needed?
  template <typename T> friend class SourceProxy;
  template <typename T> friend class SinkProxy;

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_SOURCEBASE_H
