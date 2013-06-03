/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_IOTYPEWRAPPERS_H
#define ESSENTIA_IOTYPEWRAPPERS_H

#include <string>
#include "types.h"
#include "sourcebase.h"
#include "sinkbase.h"

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
