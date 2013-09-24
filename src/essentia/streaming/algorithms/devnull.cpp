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

#include "devnull.h"
#include "../../utils/tnt/tnt.h"
using namespace std;

#define CREATE_DEVNULL(type) if (sameType(sourceType, typeid(type))) devnull = new DevNull<type>();

namespace essentia {
namespace streaming {


void connect(SourceBase& source, DevNullConnector dummy) {
  const type_info& sourceType = source.typeInfo();

  Algorithm* devnull = 0;
  CREATE_DEVNULL(int);
  CREATE_DEVNULL(Real);
  CREATE_DEVNULL(vector<Real>);
  CREATE_DEVNULL(string);
  CREATE_DEVNULL(vector<string>);
  CREATE_DEVNULL(TNT::Array2D<Real>);
  CREATE_DEVNULL(StereoSample);

  if (!devnull) throw EssentiaException("DevNull class doesn't work for type: ", nameOfType(sourceType));

  connect(source, devnull->input("data"));
}


void disconnect(SourceBase& source, DevNullConnector devnull) {
  // find dev null that this source is connected to, and disconnect it
  // note its ok we modify the list we're interating over since we return as
  // soon as we make a modification
  for (int i=0; i<int(source.sinks().size()); ++i) {
    SinkBase& sink = *(source.sinks()[i]);
    Algorithm* sinkAlg = sink.parent();

    // TODO: huh?
    if (sinkAlg->name() == "DevNull") {
      disconnect(source, sink);

      // since the DevNull is no longer connected to a network, it must be
      // manually deleted (no one should have a pointer to this instance)
      delete sinkAlg;
      return;
    }
  }

  ostringstream msg;
  msg << "the source you are disconnecting (";
  msg << source.fullName();
  msg << ") is not connected to NOWHERE";
  throw EssentiaException(msg);
}


} // namespace streaming
} // namespace essentia
