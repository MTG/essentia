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

#ifndef ESSENTIA_SINKBASE_H
#define ESSENTIA_SINKBASE_H

#include "../types.h"
#include "../connector.h"

namespace essentia {
namespace streaming {

class SinkBase;
class SinkProxyBase;
class SourceBase;
class Algorithm;


void connect(SourceBase& source, SinkBase& sink);
void disconnect(SourceBase& source, SinkBase& sink);

void attach(SinkProxyBase& proxy, SinkBase& innerSink);
void detach(SinkProxyBase& proxy, SinkBase& innerSink);


/**
 * This is the base class from which Sinks should derive. It defines the basic
 * interface a Sink should provide, such as the acquire() and release() methods,
 * and a way to get hold of one or more tokens that are waiting to be consumed.
 * It is untyped (in the sense that it doesn't know which type are the tokens),
 * but derives from TypeProxy and as such has functions to can do type-checking
 * with respect to the types of the derived Sink (which is a templated class,
 * the template being the token type). Look at the Sink implementation for more
 * information.
 */
class SinkBase : public Connector {
 protected:
  SourceBase* _source;
  ReaderID _id; // ID to use to identify this reader for the source (to know which reader is requesting tokens, etc...)

  SinkProxyBase* _sproxy;

 public:
  SinkBase(Algorithm* parent = 0, const std::string& name = "unnamed") :
    Connector(parent, name), _source(0), _sproxy(0) {}

  SinkBase(const std::string& name) :
    Connector(name), _source(0), _sproxy(0) {}

  ~SinkBase() {
    // NB: this call needs to come before the next one because _source is set by the proxy
    //     even though we're not explicitly connected to a source ourselves
    E_DEBUG(EMemory, "Deleting SinkBase " << fullName());
    if (_sproxy) essentia::streaming::detach(*_sproxy, *this);
    if (_source) essentia::streaming::disconnect(*_source, *this);
    E_DEBUG(EMemory, "Deleting SinkBase " << fullName() << "ok!");
  }

  const SourceBase* source() const { return _source; }
  SourceBase* source() { return _source; }
  virtual void setSource(SourceBase* source);

  ReaderID id() const;
  virtual void setId(ReaderID id);

  // should return a vector<TokenType>*
  virtual const void* getTokens() const = 0;

  // should return a TokenType*
  virtual const void* getFirstToken() const = 0;

 protected:
  // methods for standard connections

  // made those protected so that only our friend streaming::{dis}connect() functions can access these
  virtual void connect(SourceBase& source);
  virtual void disconnect(SourceBase& source);

  friend void connect(SourceBase& source, SinkBase& sink);
  friend void disconnect(SourceBase& source, SinkBase& sink);


  // methods for proxies

  friend void attach(SinkProxyBase& proxy, SinkBase& innerSink);
  friend void detach(SinkProxyBase& proxy, SinkBase& innerSink);

  // Note: these can't be called attach because they would be shadowed by SourceProxyBase::attach(SourceBase)
  void attachProxy(SinkProxyBase* sproxy);
  void detachProxy(SinkProxyBase* sproxy);

  // for SourceBase destructor
  friend class SourceBase;

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_SINKBASE_H
