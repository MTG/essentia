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
