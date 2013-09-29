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

#include <cstdlib>
#include <iostream>
#include "range.h"
#include "stringutil.h"

using namespace std;
using namespace essentia;

bool Everything::contains(const Parameter& param) const {
  return true;
}

Range* Range::create(const std::string& s) {
  if (s.empty()) {
    return new Everything;
  }
  else if (s[0] == '[' || s[0] == '(') {
    return new Interval(s);
  }
  else if (s[0] == '{') {
    return new Set(s);
  }
  else {
    throw EssentiaException("Invalid range");
  }
}

Interval::Interval(const string& strrange) {
  string s = strrange;

  string::size_type idx = s.find(",");
  if (idx == string::npos) {
    throw EssentiaException("Invalid interval, should contain the ',' symbol to separate both ends of the interval");
  }

  string sleft = toLower(s.substr(0, idx));
  string sright = toLower(s.substr(idx+1));

  if (sleft[0] == '[') {
    _lincluded = true;
  }
  else if (sleft[0] == '(') {
    _lincluded = false;
  }
  else {
    throw EssentiaException("Invalid interval, should contain the '[' or '(' as first character");
  }

  int endIdx = sright.size()-1;
  if (sright[endIdx] == ')') {
    _uincluded = false;
  }
  else if (sright[endIdx] == ']') {
    _uincluded = true;
  }
  else {
    throw EssentiaException("Invalid interval, should contain the ']' or ')' as last character");
  }

  sleft = sleft.substr(1);
  sright = sright.substr(0, sright.size()-1);

  if (sleft == "-inf") {
    _lbound = false;
  }
  else {
    char* ptr;
    _lbound = true;
    _lvalue = strtod(sleft.c_str(), &ptr);
    if (sleft.c_str() == ptr) {
      throw EssentiaException("Invalid interval, could not parse '", sleft, "' as a number");
    }
  }

  if (sright == "inf") {
    _ubound = false;
  }
  else {
    char* ptr;
    _ubound = true;
    _uvalue = strtod(sright.c_str(), &ptr);
    if (sright.c_str() == ptr) {
      throw EssentiaException("Invalid interval, could not parse '", sright, "' as a number");
    }
  }
}

bool Interval::contains(const Parameter& param) const {
  Real value = param.toReal();

  if (_lbound) {
    if (_lincluded && !(value >= _lvalue)) return false;
    if (!_lincluded && !(value > _lvalue)) return false;
  }

  if (_ubound) {
    if (_uincluded && !(value <= _uvalue)) return false;
    if (!_uincluded && !(value < _uvalue)) return false;
  }

  return true;
}

Set::Set(const string& srange) {
  if (srange[0] != '{') {
    throw EssentiaException("Invalid set, should contain the '{' as first character");
  }
  if (srange[srange.size() - 1] != '}') {
    throw EssentiaException("Invalid set, should contain the '}' as last character");
  }

 string s = srange.substr(1, srange.size() - 2);

  if (s.empty()) {
    throw EssentiaException("Invalid set, mustn't be empty");
  }

  vector<string> elems = tokenize(s, ",");
  _elements = set<string>(elems.begin(), elems.end());
}

bool Set::contains(const Parameter& param) const {
  return _elements.find(param.toString()) != _elements.end();
}
