/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_IOTYPEWRAPPERSIMPL_H
#define ESSENTIA_IOTYPEWRAPPERSIMPL_H

#include "iotypewrappers.h"
#include "algorithm.h"

namespace essentia {
namespace standard {


inline std::string InputBase::fullName() const {
  std::ostringstream fullname;
  fullname << (_parent ? _parent->name() : "<NoParent>") << "::" << name();
  return fullname.str();
}

template <typename Type>
void InputBase::set(const Type& data) {
  try {
    checkType<Type>();
    _data = &data;
  }
  catch (EssentiaException& e) {
    throw EssentiaException("In ", fullName(), "::set(): ", e.what());
  }
}


template <typename Type>
class Input : public InputBase {
  USE_TYPE_INFO(Type);

 public:
  const Type& get() const {
    if (!_data) {
      throw EssentiaException("In ", fullName(), "::get(): Input not bound to concrete object");
    }
    return *(Type*)_data;
  }
};



inline std::string OutputBase::fullName() const {
  std::ostringstream fullname;
  fullname << (_parent ? _parent->name() : "<NoParent>") << "::" << name();
  return fullname.str();
}

template <typename Type>
void OutputBase::set(Type& data) {
  try {
    checkType<Type>();
    _data = &data;
  }
  catch (EssentiaException& e) {
    throw EssentiaException("In ", fullName(), "::set(): ", e.what());
  }
}

template <typename Type>
class Output : public OutputBase {
  USE_TYPE_INFO(Type);

 public:
  Type& get() {
    if (!_data) {
      throw EssentiaException("In ", fullName(), "::set(): Output not bound to concrete object");
    }
    return *(Type*)_data;
  }
};


} // namespace standard
} // namespace essentia

#endif // ESSENTIA_IOTYPEWRAPPERSIMPL_H
