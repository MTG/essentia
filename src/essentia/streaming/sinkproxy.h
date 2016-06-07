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

#ifndef ESSENTIA_SINKPROXY_H
#define ESSENTIA_SINKPROXY_H

#include "sourcebase.h"
#include "multiratebuffer.h"

namespace essentia {
namespace streaming {


class SinkProxyBase : public SinkBase {
 protected:
  SinkBase* _proxiedSink;

 public:
  SinkProxyBase(Algorithm* parent = 0, const std::string& name = "unnamed") :
    SinkBase(parent, name), _proxiedSink(0) {}

  SinkProxyBase(const std::string& name) : SinkBase(name), _proxiedSink(0) {}

  ~SinkProxyBase() {
    E_DEBUG(EMemory, "Deleting SinkProxy " << fullName());
    if (_proxiedSink) essentia::streaming::detach(*this, *_proxiedSink);
  }


  const void* buffer() const {
    if (!_source)
      throw EssentiaException("SinkProxy ", fullName(), " is not currently connected to another Source");

    return _source->buffer();
  }

  void* buffer() {
    if (!_source)
      throw EssentiaException("SinkProxy ", fullName(), " is not currently connected to another Source");

    return _source->buffer();
  }

  void setId(ReaderID id) {
    SinkBase::setId(id);
    if (_proxiedSink) _proxiedSink->setId(id);
  }

  void setSource(SourceBase* source) {
    SinkBase::setSource(source);
    if (_proxiedSink) _proxiedSink->setSource(source);
  }


  //---- StreamConnector interface hijacking for proxies ----------------------------------------//

  inline void acquire() { StreamConnector::acquire(); }

  virtual bool acquire(int n) {
    throw EssentiaException("Cannot acquire for SinkProxy ", fullName(), ": you need to call acquire() on the Sink which is proxied by it");
  }

  virtual int acquireSize() const {
    if (!_proxiedSink)
      throw EssentiaException("Cannot call ::acquireSize() on SinkProxy ", fullName(), " because it is not attached");

    return _proxiedSink->acquireSize();
  }

  inline void release() { StreamConnector::release(); }

  virtual void release(int n) {
    throw EssentiaException("Cannot release for SinkProxy ", fullName(), ": you need to call release() on the Sink which is proxied by it");
  }

  virtual int releaseSize() const {
    if (!_proxiedSink)
      throw EssentiaException("Cannot call ::releaseSize() on SinkProxy ", fullName(), " because it is not attached");

    return _proxiedSink->releaseSize();
  }

  //---------------------------------------------------------------------------------------------//


  // TODO: deprecate (?)
  void updateProxiedSink() {
    if (!_proxiedSink) return;

    E_DEBUG(EConnectors, "  " << fullName() << "::updateProxiedSink: " << _proxiedSink->fullName()
            << "::setSource(" << (_source ? _source->fullName() : "0") << ")");
    _proxiedSink->setSource(_source);
    // for this to work we need to have called Source::connect(Sink) before Sink::connect(Source)
    // we should always get the same id as this sink
    E_DEBUG(EConnectors, "  " << fullName() << "::updateProxiedSink: " << _proxiedSink->fullName()
            << "::setId(" << _id << ")");
    _proxiedSink->setId(_id);

    // propagate (explicitly instead of implicit recursive with virtual func, dirty but works ok atm)
    SinkProxyBase* psink = dynamic_cast<SinkProxyBase*>(_proxiedSink);
    if (psink) {
      E_DEBUG(EConnectors, "  SinkProxy::updateProxiedSink: " << psink->fullName() << "::updateProxiedSink()");
      psink->updateProxiedSink();
    }
  }

  void detach() {
    if (_proxiedSink) essentia::streaming::detach(*this, *_proxiedSink);
  }


protected:
  /**
   * Set to which SourceBase we should proxy the SourceBase calls
   */
  void attach(SinkBase* sink) {
    // TODO: make sure the sink we're attaching to is not connected to anyone
    checkSameTypeAs(*sink);

    if (_proxiedSink) {
      // make sure the sink we're attaching to is not connected to anyone
      // Note: this doesn't prevent to connect it after being attached, but that
      //       already limits possible damage
      std::ostringstream msg;
      msg << "Could not attach SinkProxy " << fullName() << " to " << sink->fullName()
          << " because it is already attached to " << _proxiedSink->fullName();
      throw EssentiaException(msg);
    }

    E_DEBUG(EConnectors, "  SinkProxy::attach: " << fullName() << "::_proxiedSink = " << sink->fullName());
    _proxiedSink = sink;
  }

  void detach(SinkBase* sink) {
    if (sink != _proxiedSink) {
      E_WARNING("Cannot detach SinkProxy " << fullName() << " from " << sink->fullName() << " as they are not attached");
      return;
    }

    E_DEBUG(EConnectors, "  SinkProxy::detach: " << fullName() << "::_proxiedSink = 0");
    _proxiedSink = 0;

    // look inside the source(proxy) to see whether we need to remove an ID or not
    // -> no, because the source proxy only knows about the sink proxy as a normal sink,
    //    it is still connected and the reader ID is still valid. Whenever we attach a new sink
    //    to this proxy, it will reuse the reader ID
    /*
    if (_source) {
      _source->disconnect(*this);
    }
    */
  }

  friend void attach(SinkProxyBase& proxy, SinkBase& innerSink);
  friend void detach(SinkProxyBase& proxy, SinkBase& innerSink);

};

template <typename TokenType>
class Source;

template <typename TokenType>
class SinkProxy : public SinkProxyBase {
  USE_TYPE_INFO(TokenType);

 public:

  SinkProxy(Algorithm* parent = 0, const std::string& name = "unnamed") :
    SinkProxyBase(parent, name) {}

  SinkProxy(const std::string& name) : SinkProxyBase(name) {}



  //---- Buffer access methods ----------------------------------------//

  const MultiRateBuffer<TokenType>& buffer() const {
    return *static_cast<const MultiRateBuffer<TokenType>*>(SinkProxyBase::buffer());
  }

  MultiRateBuffer<TokenType>& buffer() {
    return *static_cast<MultiRateBuffer<TokenType>*>(SinkProxyBase::buffer());
  }


  void connect(SourceBase& source) {
    checkSameTypeAs(source);
    if (_source)
      throw EssentiaException("You cannot connect more than one Source to a Sink: ", fullName());

    _source = &source;
    E_DEBUG(EConnectors, "SinkProxy: sink " << fullName() << " now has source " << source.fullName());
    // for this to work we need to have called Source::connect(Sink) before Sink::connect(Source) so that
    // we already have an ID in the buffer readers' list
    updateProxiedSink();
  }


  void disconnect(SourceBase& source) {
    _source = 0;
    _proxiedSink->setSource(0);
  }

  virtual const void* getTokens() const {
    throw EssentiaException("Cannot get tokens for SinkProxy ", fullName(),
                            ": you need to call getTokens() on the Sink which is proxied by it");
  }

  virtual const void* getFirstToken() const {
    throw EssentiaException("Cannot get first token for SinkProxy ", fullName(),
                            ": you need to call getFirstToken() on the Sink which is proxied by it");
  }


  virtual int available() const {
    return buffer().availableForRead(_id);
  }

  virtual void reset() {}

};

inline void attach(SinkProxyBase& proxy, SinkBase& innerSink) {
  E_DEBUG(EConnectors, "Attaching SinkProxy " << proxy.fullName() << " to " << innerSink.fullName());
  // check types here to have a more informative error message in case it fails
  if (!sameType(proxy, innerSink)) {
    std::ostringstream msg;
    msg << "Cannot attach SinkProxy " << proxy.fullName() << " (type: " << nameOfType(proxy) << ") to "
        << innerSink.fullName() << " (type: " << nameOfType(innerSink) << ")";
    throw EssentiaException(msg);
  }
  proxy.attach(&innerSink);
  innerSink.attachProxy(&proxy);
}


inline void operator>>(SinkProxyBase& proxy, SinkBase& innerSink) {
  attach(proxy, innerSink);
}

inline void detach(SinkProxyBase& proxy, SinkBase& innerSink) {
  E_DEBUG(EConnectors, "Detaching SinkProxy " << proxy.fullName() << " from " << innerSink.fullName());
  proxy.detach(&innerSink);
  innerSink.detachProxy(&proxy);
}


} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_SINKPROXY_H
