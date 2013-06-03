/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
