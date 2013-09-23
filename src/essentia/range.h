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

#ifndef ESSENTIA_RANGE_H
#define ESSENTIA_RANGE_H

#include "parameter.h"
#include <set>

namespace essentia {

class ESSENTIA_API Range {
 public:
  virtual ~Range() {}
  static Range* create(const std::string& s);
  virtual bool contains(const Parameter& param) const = 0;
};

class Everything : public Range {
 public:
  virtual bool contains(const Parameter& param) const;
};

class Interval : public Range {
 public:
  Interval(const std::string& s);
  virtual ~Interval() {}
  virtual bool contains(const Parameter& param) const;

 protected:
  bool _lbound, _ubound;       // whether we have a lower/upper bound (ie: not infinite)
  bool _lincluded, _uincluded; // whether these bounds are inclusive or exclusive
  Real _lvalue, _uvalue;       // the boundary values
};

class Set : public Range {
 public:
  Set(const std::string& s);
  virtual ~Set() {}
  virtual bool contains(const Parameter& param) const;
  void test() const;

 protected:
  std::set<std::string> _elements;
  std::string s;
};

} // namespace essentia

#endif // ESSENTIA_RANGE_H
