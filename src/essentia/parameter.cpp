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

#include "parameter.h"

using namespace std;

namespace essentia {

string Parameter::toLower() const {
  if (!_configured)
    throw EssentiaException("Parameter: parameter has not been configured yet (ParamType=", _type, ")");

  string result = toString();
  for (int i=0; i<(int)result.size(); ++i) {
    result[i] = tolower(result[i]);
  }
  return result;
}

Parameter::Parameter(const Parameter& p) {
  *this = p;
}


Parameter::Parameter(const Parameter* p) {
  *this = *p;
}

void Parameter::clear() {
  // delete all the values from vector
  for (int i=0; i<(int)_vec.size(); i++) delete _vec[i];
  _vec.clear();

  // delete all the values from map
  for (map<string, Parameter*>::const_iterator it = _map.begin();
       it != _map.end();
       ++it) {
    delete it->second;
  }
  _map.clear();
}

Parameter& Parameter::operator=(const Parameter& p) {
  _type = p._type;
  _configured = p._configured;
  _ssamp = p._ssamp;
  _str = p._str;
  _real = p._real;
  _boolean = p._boolean;

  clear();

  for (map<string, Parameter*>::const_iterator it = p._map.begin();
       it != p._map.end();
       ++it) {
    _map[it->first] = new Parameter(*(it->second));
  }

  _vec.resize(p._vec.size());
  for (int i=0; i<(int)_vec.size(); ++i) {
    _vec[i] = new Parameter(*(p._vec[i]));
  }

  return *this;
}

Parameter::~Parameter() {
  clear();
}


string Parameter::toString(int precision) const {
  if (!_configured) {
    throw EssentiaException("Parameter: parameter has not been configured yet (ParamType=", _type, ")");
  }

  ostringstream result;
  result.precision(precision);

  switch (_type) {
    case STRING:
      result << _str;
      break;

    case REAL:
      result << _real;
      break;

    case BOOL:
      // this is a hack because when you output a boolean to a ostringstream it
      // writes a "1" or a "0"
      return _boolean ? "true" : "false";

    case INT:
      result << (int)_real;
      break;

    case STEREOSAMPLE:
      result << "{left: " << _ssamp.left() << ", right: " << _ssamp.right() << "}";
      break;

    case VECTOR_REAL:
    case VECTOR_STRING:
    case VECTOR_BOOL:
    case VECTOR_INT:
    case VECTOR_STEREOSAMPLE:
    case VECTOR_VECTOR_REAL:
    case VECTOR_VECTOR_STRING:
    case MATRIX_REAL:
    case VECTOR_MATRIX_REAL:
    case VECTOR_VECTOR_STEREOSAMPLE:
      result << "[";
      if (!_vec.empty()) {
        result << *(_vec[0]);
        for (int i=1; i<(int)_vec.size(); ++i)
          result << ", " << *(_vec[i]);
      }
      result << "]";
      break;

    case MAP_VECTOR_REAL:
    case MAP_VECTOR_STRING:
    case MAP_VECTOR_INT:
      result << "{";
      for (map<string, Parameter*>::const_iterator it = _map.begin();
           it != _map.end();
           ++it) {
        if (it != _map.begin()) result << ", ";
        result << it->first << ": " << *(it->second);
      }
      result << "}";
     break;

    case UNDEFINED:
      result << "__undefined";
      break;

    default:
      ostringstream msg;
      msg << "Parameter: cannot convert parameter (type=" << _type;
      msg << ") to a string";
      throw EssentiaException(msg.str());
  }

  return result.str();
}


bool Parameter::operator==(const Parameter& p) const {
  if (_type != p._type || _configured != p._configured) return false;

  if (!_configured && !p._configured) return true;

  switch (_type) {
    case UNDEFINED: return false;

    case STRING: return _str == p._str;
    case REAL:   return _real == p._real;
    case BOOL:   return _boolean == p._boolean;
    case INT:    return (int)_real == (int)p._real;

    case STEREOSAMPLE:
      return _ssamp.left() == p._ssamp.left() &&
             _ssamp.right() == p._ssamp.right();

    case VECTOR_REAL:
    case VECTOR_STRING:
    case VECTOR_BOOL:
    case VECTOR_INT:
    case VECTOR_STEREOSAMPLE:
    case VECTOR_VECTOR_REAL:
    case VECTOR_VECTOR_STRING:
    case MATRIX_REAL:
    case VECTOR_MATRIX_REAL:
    case VECTOR_VECTOR_STEREOSAMPLE:
      if (_vec.size() != p._vec.size()) {
        return false;
      }

      for (int i=0; i<int(_vec.size()); ++i) {
        if ( *(_vec[i]) != *(p._vec[i]) ) {
          return false;
        }
      }

      return true;

    case MAP_VECTOR_REAL:
    case MAP_VECTOR_STRING:
    case MAP_REAL:
      if (_map.size() != p._map.size())
        return false;

      for (map<string, Parameter*>::const_iterator i = _map.begin();
           i != _map.end();
           ++i) {
        if (p._map.count(i->first) == 0)
          return false;

        if ( *(i->second) != *(p._map.find(i->first)->second) ) {
          return false;
        }
      }

      return true;

    default:
      throw EssentiaException("Parameter: the == operator does not support parameter type: ", _type);
  }
}


bool Parameter::operator!=(const Parameter& p) const {
  return !(*this == p);
}


void ParameterMap::add(const string& name, const Parameter& value) {
  std::pair<ParameterMap::iterator, bool> inserted = this->insert(name, value);
  // if there already was a value with the same key, we have to overwrite it
  if (!inserted.second) inserted.first->second = value;
}


const Parameter& ParameterMap::operator[](const string& key) const {
  return ParameterMapBase::operator[](key);
}

Parameter& ParameterMap::operator[](const string& key) {
  return ParameterMapBase::operator[](key);
}

ostream& operator<<(ostream& out, const Parameter& p) {
  // special case for strings. these need to be output enclosed in double
  // quotes, and any double quotes and backslashes inside them need to be escaped
  if (p.type() == Parameter::STRING) {
    out << "\"";
    string unescaped = p.toString();
    for (int i=0; i<int(unescaped.size()); ++i) {
      // escape double quotation marks
      if (unescaped[i] == '\"' || unescaped[i] == '\\') out << "\\";
      out << unescaped[i];
    }
    return out << "\"";
  }
  else {
    return out << p.toString();
  }
}

ostream& operator<<(ostream& out, const ParameterMap& m) {
  out << '{';
  if (!m.empty()) {
    ParameterMap::const_iterator it = m.begin();
    out << " '" << it->first << "':'" << it->second << "'";
    ++it;
    for (; it != m.end(); ++it) {
      out << ", '" << it->first << "':'" << it->second << "'";
    }
  }
  return out << " }";
}

ostream& operator<<(ostream& out, const Parameter::ParamType& t) {
  switch (t) {
    case Parameter::UNDEFINED:            return out << "UNDEFINED";
    case Parameter::STRING:               return out << "STRING";
    case Parameter::REAL:                 return out << "REAL";
    case Parameter::BOOL:                 return out << "BOOL";
    case Parameter::INT:                  return out << "INT";
    case Parameter::STEREOSAMPLE:         return out << "STEREOSAMPLE";
    case Parameter::VECTOR_REAL:          return out << "VECTOR_REAL";
    case Parameter::VECTOR_STRING:        return out << "VECTOR_STRING";
    case Parameter::VECTOR_BOOL:          return out << "VECTOR_BOOL";
    case Parameter::VECTOR_INT:           return out << "VECTOR_INT";
    case Parameter::VECTOR_STEREOSAMPLE:  return out << "VECTOR_STEREOSAMPLE";
    case Parameter::VECTOR_VECTOR_REAL:   return out << "VECTOR_VECTOR_REAL";
    case Parameter::VECTOR_VECTOR_STRING: return out << "VECTOR_VECTOR_STRING";
    case Parameter::VECTOR_VECTOR_STEREOSAMPLE: return out << "VECTOR_VECTOR_STEREOSAMPLE";
    case Parameter::MAP_VECTOR_REAL:      return out << "MAP_VECTOR_REAL";
    case Parameter::MAP_VECTOR_STRING:    return out << "MAP_VECTOR_STRING";
    case Parameter::MATRIX_REAL:          return out << "MATRIX_REAL";
    case Parameter::VECTOR_MATRIX_REAL:   return out << "VECTOR_MATRIX_REAL";
    case Parameter::MAP_REAL:             return out << "MAP_REAL";
    case Parameter::MAP_VECTOR_INT:       return out << "MAP_VECTOR_INT";
    default:                              return out << "ParamType(" << int(t) << ")";
  }
}

} // namespace essentia
