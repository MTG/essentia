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

#ifndef ESSENTIA_IOTYPEWRAPPERS_H
#define ESSENTIA_IOTYPEWRAPPERS_H

#include <string>
#include "types.h"
#include "streaming/sourcebase.h"
#include "streaming/sinkbase.h"

namespace essentia {
namespace standard {


class Algorithm;


class ESSENTIA_API InputBase : public TypeProxy {

 protected:
  Algorithm* _parent;
  friend class Algorithm;

 public:
  InputBase() : _parent(0), _data(0) {}

  std::string fullName() const;

  // implementation in iotypewrappers_impl.h
  template <typename Type>
  void set(const Type& data);

  void setSinkFirstToken(streaming::SinkBase& sink) {
    checkSameTypeAs(sink);
    _data = sink.getFirstToken();
  }

  void setSinkTokens(streaming::SinkBase& sink) {
    checkVectorSameTypeAs(sink);
    _data = sink.getTokens();
  }

 protected:
  const void* _data;

};


class ESSENTIA_API OutputBase : public TypeProxy {

 protected:
  Algorithm* _parent;
  friend class Algorithm;

 public:
  OutputBase() : _parent(0), _data(0) {}

  std::string fullName() const;

  // implementation in iotypewrappers_impl.h
  template <typename Type>
  void set(Type& data);

  void setSourceFirstToken(streaming::SourceBase& source) {
    checkSameTypeAs(source);
    _data = source.getFirstToken();
  }

  void setSourceTokens(streaming::SourceBase& source) {
    checkVectorSameTypeAs(source);
    _data = source.getTokens();
  }

 protected:
  void* _data;

};


} // namespace standard
} // namespace essentia

#endif // ESSENTIA_IOTYPEWRAPPERS_H
