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

#include "poolstorage.h"
using namespace std;

#define CREATE_POOL_STORAGE(type)\
  if (sameType(sourceType, typeid(type))) {\
     ps = new PoolStorage<type>(&pool, descriptorName, setSingle);\
  }

namespace essentia {
namespace streaming {

void connect(SourceBase& source, Pool& pool, const string& descriptorName, bool setSingle) {

  const type_info& sourceType = source.typeInfo();

  Algorithm* ps = 0;
  //if (sameType(sourceType, typeid(Real))) ps = new PoolStorage<Real>(&pool, name);

  CREATE_POOL_STORAGE(Real);
  CREATE_POOL_STORAGE(string);
  CREATE_POOL_STORAGE(vector<string>);
  CREATE_POOL_STORAGE(TNT::Array2D<Real>);
  CREATE_POOL_STORAGE(StereoSample);
  CREATE_POOL_STORAGE(vector<Real>);

  // convert int to Real
  if (sameType(sourceType, typeid(int))) ps = new PoolStorage<int, Real>(&pool, descriptorName, setSingle);

  if (!ps) throw EssentiaException("Pool Storage doesn't work for type: ", nameOfType(sourceType));

  try {
    connect(source, ps->input("data"));
  }
  catch (EssentiaException& e) {
    std::ostringstream msg;
    msg << "While connecting " << source.fullName()
        << " to Pool[" << descriptorName << "]:\n"
        << e.what();
    throw EssentiaException(msg);
  }
}

void connect(SourceBase& source, Pool& pool, const string& descriptorName) {
  bool setSingle = false;
  if (source.releaseSize() == 0) setSingle = true;

  connect(source, pool, descriptorName, setSingle);
}

void connectSingleValue(SourceBase& source, Pool& pool, const string& descriptorName) {
  connect(source, pool, descriptorName, true);
}


#define GET_POOLSTORAGE_PROPERTIES(type) \
  if (sameType(sourceType, typeid(type))) { \
    p = ((PoolStorage<type>*)sinkAlg)->pool(); \
    dname = ((PoolStorage<type>*)sinkAlg)->descriptorName(); \
  }


void disconnect(SourceBase& source, Pool& pool, const string& descriptorName) {
  // find pool storage that this source is connected to (one that matches the
  // pool and the name), and disconnect it
  for (int i=0; i<int(source.sinks().size()); ++i) {
    SinkBase& sink = *(source.sinks()[i]);
    Algorithm* sinkAlg = sink.parent();

    if (sinkAlg->name() == "PoolStorage") {
      const type_info& sourceType = source.typeInfo();
      Pool* p;
      string dname;

      GET_POOLSTORAGE_PROPERTIES(Real)
      else GET_POOLSTORAGE_PROPERTIES(string)
      else GET_POOLSTORAGE_PROPERTIES(vector<string>)
      else GET_POOLSTORAGE_PROPERTIES(vector<Real>)
      else GET_POOLSTORAGE_PROPERTIES(TNT::Array2D<Real>)
      else GET_POOLSTORAGE_PROPERTIES(StereoSample)
      else if (sameType(sourceType, typeid(int))) {
        p = ((PoolStorage<Real>*)sinkAlg)->pool();
        dname = ((PoolStorage<Real>*)sinkAlg)->descriptorName();
      }
      else {
        ostringstream msg;
        msg << "internal error: it seems that a source (";
        msg << source.parent()->name() << "::" << source.name();
        msg << ") with an invalid type has been connected to a Pool, this shouldn't happen";
        throw EssentiaException(msg);
      }

      if (p == &pool && dname == descriptorName) {
        disconnect(source, sink);

        // since the PoolStorage is no longer connected to a network, it must be
        // manually deleted (no one should have a pointer to this instance)
        delete sinkAlg;
        return;
      }
    }
  }

  ostringstream msg;
  msg << "the source you are disconnecting (";
  msg << source.parent()->name() << "::" << source.name();
  msg << ") is not connected to a Pool";
  throw EssentiaException(msg);
}


} // namespace streaming
} // namespace essentia
