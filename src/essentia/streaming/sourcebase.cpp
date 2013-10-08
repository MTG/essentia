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

#include "sourcebase.h"
#include "sinkbase.h"
#include "sourceproxy.h"
#include "essentiautil.h" // for contains
using namespace std;

namespace essentia {
namespace streaming {


SourceBase::~SourceBase() {
  E_DEBUG(EMemory, "Deleting SourceBase " << fullName());
  if (_sproxy) {
    // do not do the streaming::detach, because this object is already a SourceBase
    // only (ie: not a Source or SourceProxy anymore), so we can't call any of the
    // reader IDs methods which are pure virtual now...
    //essentia::streaming::detach(*this, *_sproxy);
    _sproxy->detach(this);
  }

  // TODO: here we want to set the source of those sinks which are connected direcxtly to us to 0,
  //       but leave the source of those sinks that are connected through the (optional) proxy
  for (int i=0; i<(int)_sinks.size(); i++) {
    if (!_sproxy ||
        (_sproxy && !contains(_sproxy->sinks(), _sinks[i]))) {
      E_DEBUG(EMemory, fullName() << "::dtor : disconnect directly connected sink " << i << " - " << _sinks[i]->fullName());
      _sinks[i]->disconnect(*this);
    }
  }
}

void SourceBase::connect(SinkBase& sink) {
  checkSameTypeAs(sink);
  // do not connect twice the sink
  if (contains(_sinks, &sink)) {
    E_WARNING(this->fullName() << " is already connected to " << sink.fullName());
    return;
  }

  ReaderID id = addReader();

  E_DEBUG(EConnectors, "  SourceBase::connect: id = AddReader(); " << sink.fullName() << "::setId(" << id << ")");
  sink.setId(id);
  _sinks.push_back(&sink);
}


void SourceBase::disconnect(SinkBase& sink) {
  // find sink and remove it
  bool found = false;
  int i = 0;
  for (; i<(int)_sinks.size(); i++) {
    if (_sinks[i] == &sink) {
      E_DEBUG(EConnectors, "  SourceBase::disconnect: removeReader(" << i << "): " << sink.fullName());
      removeReader(i);
      _sinks.erase(_sinks.begin() + i);
      found = true;
      break;
    }
  }

  if (!found) {
    E_WARNING(this->fullName() << " was not connected to " << sink.fullName());
    return;
  }

  // all sinks after the one we just removed must have their readerID decreased by one
  for (; i<(int)_sinks.size(); i++) {
    _sinks[i]->setId(i);
  }
}


void SourceBase::attachProxy(SourceProxyBase* sproxy) {
  // attach all the sinks from the proxied source
  checkSameTypeAs(*sproxy);

  // disconnect previous proxy
  // TODO: disconnect automatically or throw exception telling to disconnect explicitly
  if (_sproxy) {
    // TODO: disconnect all of them
    E_WARNING("ARGLLLLLL");
  }


  E_DEBUG(EConnectors, "  SourceBase::attachProxy: " << fullName() << "::_sproxy = " << sproxy->fullName());
  _sproxy = sproxy;

  E_DEBUG(EConnectors, "  SourceBase::attachProxy: " << fullName() << "::connectAllSinks");
  const vector<SinkBase*>& sinks = sproxy->sinks();
  for (int i=0; i<(int)sinks.size(); i++) {
    connect(*sinks[i]);
  }
}


void SourceBase::detachProxy(SourceProxyBase* sproxy) {
  if (sproxy != _sproxy) {
    E_WARNING("Cannot detach " << fullName() << " from SourceProxy " << sproxy->fullName() << " as they are not attached");
    return;
  }

  // first remove all readers that came from that sourceproxy
  for (int i=0; i<(int)sproxy->sinks().size(); i++) {
    disconnect(*sproxy->sinks()[i]);
  }

  E_DEBUG(EConnectors, "  SourceBase::detachProxy: " << fullName() << "::_sproxy = 0");
  _sproxy = 0;
}

const vector<SinkBase*>& SourceBase::proxiedSinks() const {
  return _sproxy->sinks();
}


} // namespace streaming
} // namespace essentia
