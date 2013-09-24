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

#include "sinkbase.h"
#include "sinkproxy.h"
#include "sourcebase.h"

namespace essentia {
namespace streaming {

ReaderID SinkBase::id() const {
  // NOTE: if this sink is connected to a sourceproxy, it will have a _source set, but the ID is still invalid...
  if (_source) return _id;
  else
    throw EssentiaException("Undefined reader ID for sink ", fullName());
}

void SinkBase::setId(ReaderID id) {
 _id = id;
}

void SinkBase::setSource(SourceBase* source) {
  E_DEBUG(EConnectors, fullName() << "::setSource(" << (source ? source->fullName() : "0") << ")");
  _source = source;
}


void SinkBase::connect(SourceBase& source) {
  checkSameTypeAs(source);
  if (_source)
    throw EssentiaException("You cannot connect more than one Source to a Sink: ",
                            fullName(), " is already connected to ", _source->fullName());

  if (_sproxy)
    throw EssentiaException("You cannot connect a Source to a Sink which is already attached to a SinkProxy: ",
                            fullName(), " is connected to proxy ", _sproxy->fullName());

  E_DEBUG(EConnectors, "  SinkBase::connect: " << fullName() << "::_source = " << source.fullName());
  _source = &source;
}

// NB: do not do anything if we're not actually connected to the given source
void SinkBase::disconnect(SourceBase& source) {
  if (_source != &source) {
    E_WARNING("Cannot disconnect " << this->fullName() << " from " << source.fullName() << " as they are not connected");
    return;
  }

  E_DEBUG(EConnectors, "  SinkBase::disconnect: " << fullName() << "::_source = 0");
  setSource(0);
}


void SinkBase::attachProxy(SinkProxyBase* sproxy) {
  checkSameTypeAs(*sproxy);

  if (_source)
    throw EssentiaException("You cannot attach a SinkProxy to a Sink which is already connected: ",
                            fullName(), " is already connected to ", _source->fullName());

  if (_sproxy)
    throw EssentiaException("You cannot attach a SinkProxy to a Sink which is already attached to a SinkProxy: ",
                            fullName(), " is attached to proxy ", _sproxy->fullName());

  E_DEBUG(EConnectors, "  SinkBase::attachProxy: " << fullName() << "::_sproxy = " << sproxy->fullName());
  _sproxy = sproxy;
  E_DEBUG(EConnectors, "  SinkBase::attachProxy: " << sproxy->fullName() << "::updateProxiedSink()");
  _sproxy->updateProxiedSink();
}

void SinkBase::detachProxy(SinkProxyBase* sproxy) {
  // TODO: verify me
  if (sproxy != _sproxy) {
    E_WARNING("Cannot detach " << fullName() << " from SinkProxy " << sproxy->fullName() << " as they are not attached");
    return;
  }

  E_DEBUG(EConnectors, "  SinkBase::detachProxy: " << fullName() << "::_sproxy = 0");
  _sproxy = 0;
  E_DEBUG(EConnectors, "  SinkBase::detachProxy: " << fullName() << "::_source = 0");
  setSource(0); // also set source to 0 because the proxy set it for us when we attached, but now we're all alone

}

} // namespace streaming
} // namespace essentia
