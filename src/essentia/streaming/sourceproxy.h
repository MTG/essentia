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

#ifndef ESSENTIA_SOURCEPROXY_H
#define ESSENTIA_SOURCEPROXY_H

#include <vector>
#include "sourcebase.h"
#include "multiratebuffer.h"
#include "sink.h"
#include "essentiautil.h"


namespace essentia {
namespace streaming {

/**
 * Non template base class for the proxy source, contains a pointer to the actual
 * Source being proxied.
 */
class SourceProxyBase : public SourceBase {
 protected:
  SourceBase* _proxiedSource;

 public:
  SourceProxyBase(Algorithm* parent = 0, const std::string& name = "unnamed") :
    SourceBase(parent, name), _proxiedSource(0) {}

  SourceProxyBase(const std::string& name) : SourceBase(name), _proxiedSource(0) {}

  ~SourceProxyBase() {
    E_DEBUG(EMemory, "Deleting SourceProxy " << fullName());
    if (_proxiedSource) essentia::streaming::detach(*_proxiedSource, *this);
  }

  SourceBase* proxiedSource() { return _proxiedSource; }


  //---- Buffer access methods ----------------------------------------//

  const void* buffer() const {
    if (!_proxiedSource)
      throw EssentiaException("SourceProxy ", fullName(), " is not currently attached to another Source");

    return _proxiedSource->buffer();
  }

  void* buffer() {
    if (!_proxiedSource)
      throw EssentiaException("SourceProxy ", fullName(), " is not currently attached to another Source");

    return _proxiedSource->buffer();
  }

  virtual void setBufferType(BufferUsage::BufferUsageType type) {
    _proxiedSource->setBufferType(type);
  }

  virtual BufferInfo bufferInfo() const {
    return _proxiedSource->bufferInfo();
  }

  virtual void setBufferInfo(const BufferInfo& info) {
    _proxiedSource->setBufferInfo(info);
  }


  //---- StreamConnector interface hijacking for proxies ----------------------------------------//

  inline void acquire() { StreamConnector::acquire(); }

  virtual bool acquire(int n) {
    throw EssentiaException("Cannot acquire for SourceProxy ", fullName(), ": you need to call acquire() on the Source which is proxied by it");
  }

  virtual int acquireSize() const {
    if (!_proxiedSource)
      throw EssentiaException("Cannot call ::acquireSize() on SourceProxy ", fullName(), " because it is not attached");

    return _proxiedSource->acquireSize();
  }

  inline void release() { StreamConnector::release(); }

  virtual void release(int n) {
    throw EssentiaException("Cannot release for SourceProxy ", fullName(), ": you need to call release() on the Source which is proxied by it");
  }

  virtual int releaseSize() const {
    if (!_proxiedSource)
      throw EssentiaException("Cannot call ::releaseSize() on SourceProxy ", fullName(), " because it is not attached");

    return _proxiedSource->releaseSize();
  }

  //---------------------------------------------------------------------------------------------//


  void detach() {
    if (_proxiedSource) essentia::streaming::detach(*_proxiedSource, *this);
  }

  virtual void connect(SinkBase& sink) {
    SourceBase::connect(sink);
    if (_proxiedSource) {
      E_DEBUG(EConnectors, "  SourceProxy " << fullName() << "::connect: " << _proxiedSource->fullName()
            << "::connect(" << sink.fullName() << ")");

      _proxiedSource->connect(sink);
    }
  }

  virtual void disconnect(SinkBase& sink) {
    SourceBase::disconnect(sink);
    if (_proxiedSource) {
      E_DEBUG(EConnectors, "  SourceProxy " << fullName() << "::disconnect: " << _proxiedSource->fullName()
            << "::disconnect(" << sink.fullName() << ")");
      _proxiedSource->disconnect(sink);
    }
  }

 protected:
  /**
   * Set to which SourceBase we should proxy the SourceBase calls
   */
  void attach(SourceBase* source) {
    checkSameTypeAs(*source);

    if (_proxiedSource) {
      std::ostringstream msg;
      msg << "Could not attach SourceProxy " << fullName() << " to " << source->fullName()
          << " because it is already attached to " << _proxiedSource->fullName();
      throw EssentiaException(msg);
    }

    E_DEBUG(EConnectors, "  SourceProxy::attach: " << fullName() << "::_proxiedSource = " << source->fullName());
    _proxiedSource = source;
  }

  friend void attach(SourceBase& innerSource, SourceProxyBase& proxy);
  friend void detach(SourceBase& innerSource, SourceProxyBase& proxy);

  void detach(SourceBase* source) {
    if (source != _proxiedSource) {
      E_WARNING("Cannot detach SourceProxy " << fullName() << " from " << source->fullName() << " as they are not attached");
      return;
    }

    E_DEBUG(EConnectors, "  SourceProxy::detach: " << fullName() << "::_proxiedSource = 0");
    _proxiedSource = 0;
  }

  // for SourceBase destructor
  friend class SourceBase;
};


template<typename TokenType>
class SourceProxy : public SourceProxyBase {
  USE_TYPE_INFO(TokenType);

 public:

  SourceProxy(Algorithm* parent = 0, const std::string& name = "unnamed") :
    SourceProxyBase(parent, name) {}

  SourceProxy(const std::string& name) : SourceProxyBase(name) {}

  ~SourceProxy() {}


 public:

  //---- Buffer access methods ----------------------------------------//


  const MultiRateBuffer<TokenType>& typedBuffer() const {
    return *static_cast<const MultiRateBuffer<TokenType>*>(buffer());
  }

  MultiRateBuffer<TokenType>& typedBuffer() {
    return *static_cast<MultiRateBuffer<TokenType>*>(buffer());
  }


  //---- Connect methods ----------------------------------------------//

  ReaderID addReader() {
    // return some (random) value as it's gonna be overwritten as soon as we're connected to a real source somehow
    return _sinks.size();
  }

  void removeReader(ReaderID id) {
    return;
  }

  virtual void* getTokens() {
    throw EssentiaException("Cannot get tokens for SourceProxy ", fullName(),
                            ": you need to call getTokens() on the Source which is proxied by it");
  }

  virtual void* getFirstToken() {
    throw EssentiaException("Cannot get first token for SourceProxy ", fullName(),
                            ": you need to call getFirstToken() on the Source which is proxied by it");
  }


  virtual int available() const {
    return typedBuffer().availableForWrite(false);
  }

  int totalProduced() const {
    if (!_proxiedSource)
      throw EssentiaException("Cannot call ::totalProduced() on SourceProxy ", fullName(), " because it is not attached");

    return _proxiedSource->totalProduced();
  }

  virtual void reset() {
    // NB: do not throw an exception here, it is ok to reset a non-attached SourceProxy
    if (_proxiedSource) _proxiedSource->reset();
  }

};


inline void attach(SourceBase& innerSource, SourceProxyBase& proxy) {
  E_DEBUG(EConnectors, "Attaching SourceProxy " << proxy.fullName() << " to " << innerSource.fullName());
  // check types here to have a more informative error message in case it fails
  if (!sameType(innerSource, proxy)) {
    std::ostringstream msg;
    msg << "Cannot attach " << innerSource.fullName() << " (type: " << nameOfType(innerSource)
        << ") to SourceProxy " << proxy.fullName() << " (type: " << nameOfType(proxy) << ")";
    throw EssentiaException(msg);
  }
  proxy.attach(&innerSource);
  innerSource.attachProxy(&proxy);
}

inline void operator>>(SourceBase& innerSource, SourceProxyBase& proxy) {
  attach(innerSource, proxy);
}

inline void detach(SourceBase& innerSource, SourceProxyBase& proxy) {
  E_DEBUG(EConnectors, "Detaching SourceProxy " << proxy.fullName() << " from " << innerSource.fullName());
  proxy.detach(&innerSource);
  innerSource.detachProxy(&proxy);
}


} // namespace essentia
} // namespace streaming


#endif // ESSENTIA_SOURCEPROXY_H
