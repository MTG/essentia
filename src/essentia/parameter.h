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

#ifndef ESSENTIA_PARAMETER_H
#define ESSENTIA_PARAMETER_H

#include <map>
#include <vector>
#include "types.h"
#include "utils/tnt/tnt_array2d.h"

namespace essentia {

class ESSENTIA_API Parameter {

 public:

  enum ParamType {
    UNDEFINED,

    REAL,
    STRING,
    BOOL,
    INT,
    STEREOSAMPLE,

    VECTOR_REAL,
    VECTOR_STRING,
    VECTOR_BOOL,
    VECTOR_INT,
    VECTOR_STEREOSAMPLE,

    VECTOR_VECTOR_REAL,
    VECTOR_VECTOR_STRING,
    VECTOR_VECTOR_STEREOSAMPLE,

    VECTOR_MATRIX_REAL,

    MAP_VECTOR_REAL,
    MAP_VECTOR_STRING,
    MAP_VECTOR_INT,
    MAP_REAL,

    MATRIX_REAL
  };

 private:

  ParamType _type;

  std::string _str;
  Real _real;
  bool _boolean;
  std::vector<Parameter*> _vec;
  std::map<std::string, Parameter*> _map;
  StereoSample _ssamp;
  bool _configured;

 public:

  // Constructor for just declaring type (not providing a value)
  Parameter(ParamType tp) : _type(tp), _configured(false) {}

  // Constructor for simple parameters
  #define SPECIALIZE_CTOR(valueType, paramType, mName)                         \
  Parameter (const valueType& x) : _type(paramType), _##mName(x), _configured(true) {}

  SPECIALIZE_CTOR(std::string,  STRING,       str);
  SPECIALIZE_CTOR(Real,         REAL,         real);
  SPECIALIZE_CTOR(bool,         BOOL,         boolean);
  SPECIALIZE_CTOR(int,          INT,          real);
  SPECIALIZE_CTOR(double,       REAL,         real);
  SPECIALIZE_CTOR(uint,         INT,          real);
  SPECIALIZE_CTOR(StereoSample, STEREOSAMPLE, ssamp);

  Parameter(const char* x) : _type(STRING), _str(x), _configured(true) {}

  // Constructor for vector parameters
  #define SPECIALIZE_VECTOR_CTOR(valueType, paramType)                         \
  Parameter(const std::vector<valueType>& v) : _type(paramType), _configured(true) {\
    _vec.resize(v.size());                                                     \
    for (int i=0; i<int(v.size()); ++i) { _vec[i] = new Parameter(v[i]); }     \
  }

  SPECIALIZE_VECTOR_CTOR(Real,                      VECTOR_REAL);
  SPECIALIZE_VECTOR_CTOR(std::string,               VECTOR_STRING);
  SPECIALIZE_VECTOR_CTOR(bool,                      VECTOR_BOOL);
  SPECIALIZE_VECTOR_CTOR(int,                       VECTOR_INT);
  SPECIALIZE_VECTOR_CTOR(StereoSample,              VECTOR_STEREOSAMPLE);
  SPECIALIZE_VECTOR_CTOR(std::vector<Real>,         VECTOR_VECTOR_REAL);
  SPECIALIZE_VECTOR_CTOR(std::vector<std::string>,  VECTOR_VECTOR_STRING);
  SPECIALIZE_VECTOR_CTOR(std::vector<StereoSample>, VECTOR_VECTOR_STEREOSAMPLE);
  SPECIALIZE_VECTOR_CTOR(TNT::Array2D<Real>,        VECTOR_MATRIX_REAL);

  // Constructor for map parameters
  #define SPECIALIZE_MAP_CTOR(valueType, paramType)                            \
  Parameter(const std::map<std::string, valueType>& m) : _type(paramType), _configured(true) { \
    for (std::map<std::string, valueType>::const_iterator i = m.begin();       \
         i != m.end();                                                         \
         ++i) { _map[(*i).first] = new Parameter((*i).second); }               \
  }

  SPECIALIZE_MAP_CTOR(std::vector<std::string>, MAP_VECTOR_STRING);
  SPECIALIZE_MAP_CTOR(std::vector<Real>, MAP_VECTOR_REAL);
  SPECIALIZE_MAP_CTOR(std::vector<int>, MAP_VECTOR_INT);
  SPECIALIZE_MAP_CTOR(Real, MAP_REAL);

  // Constructor for TNT::Array2D aka MATRIX parameters
  #define SPECIALIZE_MATRIX_CTOR(valueType, innerType)                         \
  Parameter(const TNT::Array2D<valueType>& mat) : _type(MATRIX_##innerType), _configured(true) { \
    _vec.resize(mat.dim1());                                                   \
    for (int i=0; i<mat.dim1(); ++i) {                                         \
      _vec[i] = new Parameter(VECTOR_##innerType);                             \
      _vec[i]->_configured = true;                                             \
      _vec[i]->_vec.resize(mat.dim2());                                        \
      for (int j=0; j<mat.dim2(); ++j) {                                       \
        _vec[i]->_vec[j] = new Parameter(mat[i][j]);                           \
      }                                                                        \
    }                                                                          \
  }

  SPECIALIZE_MATRIX_CTOR(Real, REAL)

  Parameter(const Parameter& p);

  // also define ctor with a ptr, which allows a nice trick: we can now construct
  // a Parameter from a map<string, Parameter*> which do not necessarily have the
  // same type
  Parameter(const Parameter* p);
  ~Parameter();

  void clear();

  Parameter& operator=(const Parameter& p);
  bool operator==(const Parameter& p) const;
  bool operator!=(const Parameter& p) const;
  ParamType type() const { return _type; }
  bool isConfigured() const { return _configured; }



  std::string toString(int precision = 12) const;
  std::string toLower() const;

  #define TO(fname, valueType, paramType, mName)                              \
  valueType to##fname() const {                                               \
    if (!_configured)                                                         \
      throw EssentiaException("Parameter: parameter has not been configured yet (ParamType=", _type, ")"); \
    if (_type != paramType)                                                   \
      throw EssentiaException("Parameter: parameter is not a " #valueType ", it is a ", _type); \
                                                                              \
    return (valueType)_##mName;                                               \
  }

  TO(Bool, bool, BOOL, boolean)
  TO(Double, double, REAL, real)
  TO(Float, float, REAL, real)
  TO(StereoSample, StereoSample, STEREOSAMPLE, ssamp)

  int toInt() const {
    if (!_configured)
      throw EssentiaException("Parameter: parameter has not been configured yet (ParamType=", _type, ")");
    if (_type != INT && _type != REAL)
      throw EssentiaException("Parameter: parameter is not an int nor a Real, it is a ", _type);

    return (int)_real;
  }

  // special case for toReal because it can return an int or a real
  Real toReal() const {
    if (!_configured)
      throw EssentiaException("Parameter: parameter has not been configured yet (ParamType=", _type, ")");
    if (_type != INT && _type != REAL)
      throw EssentiaException("Parameter: parameter is not an int nor a Real, it is a ", _type);

    return _real;
  }

  #define TOVECTOR(fname, valueType, paramType)                               \
  std::vector<valueType > toVector##fname() const {                           \
    if (!_configured)                                                         \
      throw EssentiaException("Parameter: parameter has not been configured yet (ParamType=", _type, ")"); \
    if (_type != paramType)                                                   \
      throw EssentiaException("Parameter: parameter is not of type: ", paramType); \
                                                                              \
    std::vector<valueType > result(_vec.size());                              \
    for (int i=0; i<(int)_vec.size(); ++i) {                                  \
      result[i] = _vec[i]->to##fname();                                       \
    }                                                                         \
    return result;                                                            \
  }

  TOVECTOR(Real, Real, VECTOR_REAL)
  TOVECTOR(String, std::string, VECTOR_STRING)
  TOVECTOR(Int, int, VECTOR_INT)
  TOVECTOR(Bool, bool, VECTOR_BOOL)
  TOVECTOR(StereoSample, StereoSample, VECTOR_STEREOSAMPLE)
  TOVECTOR(VectorReal, std::vector<Real>, VECTOR_VECTOR_REAL)
  TOVECTOR(VectorString, std::vector<std::string>, VECTOR_VECTOR_STRING)
  TOVECTOR(VectorStereoSample, std::vector<StereoSample>, VECTOR_VECTOR_STEREOSAMPLE)
  TOVECTOR(MatrixReal, TNT::Array2D<Real>, VECTOR_MATRIX_REAL)
//  TOVECTOR(MatrixInt, TNT::Array2D<int>, VECTOR_MATRIX_INT)

  #define TOMAP(fname, valueType, paramType)                                   \
  std::map<std::string, valueType > toMap##fname() const {                     \
    if (!_configured)                                                          \
      throw EssentiaException("Parameter: parameter has not been configured yet (ParamType=", _type, ")"); \
    if (_type != paramType)                                                    \
      throw EssentiaException("Parameter: parameter is not of type: ", paramType); \
                                                                               \
    std::map<std::string, valueType > result;                                  \
                                                                               \
    for (std::map<std::string, Parameter*>::const_iterator i = _map.begin();   \
         i != _map.end();                                                      \
         ++i) {                                                                \
      result[i->first] = i->second->to##fname();                               \
    }                                                                          \
                                                                               \
    return result;                                                             \
  }

  TOMAP(VectorReal, std::vector<Real>, MAP_VECTOR_REAL)
  TOMAP(VectorString, std::vector<std::string>, MAP_VECTOR_STRING)
  TOMAP(VectorInt, std::vector<int>, MAP_VECTOR_INT)
  TOMAP(Real, Real, MAP_REAL)
//  TOMAP(String, std::string, MAP_STRING)
//  TOMAP(Int, int, MAP_INT)
//  TOMAP(Bool, bool, MAP_BOOL)
//  TOMAP(StereoSample, StereoSample, MAP_STEREOSAMPLE)

  #define TOMATRIX(fname, valueType, paramType)                                \
  TNT::Array2D<valueType> toMatrix##fname() const {                            \
    if (!_configured)                                                          \
      throw EssentiaException("Parameter: parameter has not been configured yet (ParamType=", _type, ")");\
    if (_type != paramType)                                                    \
      throw EssentiaException("Parameter: parameter is not of type: ", paramType);\
    TNT::Array2D<valueType> result(_vec.size(), _vec[0]->_vec.size());         \
                                                                               \
    for (int i=0; i<result.dim1(); ++i) {                                      \
      for (int j=0; j<result.dim2(); ++j) {                                    \
        result[i][j] = _vec[i]->_vec[j]->to##fname();                          \
      }                                                                        \
    }                                                                          \
    return result;                                                             \
  }

  TOMATRIX(Real, Real, MATRIX_REAL)
//  TOMATRIX(String, std::string, MATRIX_STRING)
//  TOMATRIX(Int, int, MATRIX_INT)
//  TOMATRIX(Bool, bool, MATRIX_BOOL)

};

/**
 * A ParameterMap is a map whose keys are strings and values are Parameter, and
 * which uses a case-insensitive compare function. It also has two convenient
 * functions for adding new values in it: add() with or without a
 * default value.
 */
class ParameterMap : public EssentiaMap<std::string, Parameter, string_cmp> {

 protected:
  typedef EssentiaMap<std::string, Parameter, string_cmp> ParameterMapBase;

 public:
  void add(const std::string& name, const Parameter& value);

  const Parameter& operator[](const std::string& key) const;
  Parameter& operator[](const std::string& key);

};

std::ostream& operator<<(std::ostream& out, const Parameter& p);
std::ostream& operator<<(std::ostream& out, const ParameterMap& m);
std::ostream& operator<<(std::ostream& out, const Parameter::ParamType& t);

} // namespace essentia

#endif // ESSENTIA_PARAMETER_H
