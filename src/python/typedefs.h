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

#ifndef ESSENTIA_PYTHON_TYPEDEFS_H
#define ESSENTIA_PYTHON_TYPEDEFS_H

#include <Python.h>
#include <complex>
#define NO_IMPORT_ARRAY
#include "numpy/ndarrayobject.h"
#include "types.h"
#include "parameter.h"
#include "pytypes/pypool.h"
#include "roguevector.h"
#include "typewrapper.h"
#include "tnt/tnt.h"

// Essentia Data Type
enum Edt {
  REAL,
  STRING,
  INTEGER,
  BOOL,
  STEREOSAMPLE,
  VECTOR_REAL,
  VECTOR_STRING,
  VECTOR_COMPLEX,
  VECTOR_INTEGER,
  VECTOR_STEREOSAMPLE,
  VECTOR_BOOL,
  VECTOR_VECTOR_REAL,
  VECTOR_VECTOR_STRING,
  VECTOR_VECTOR_STEREOSAMPLE,
  MATRIX_REAL,
  VECTOR_MATRIX_REAL,
  POOL,
  MAP_VECTOR_STRING,
  MAP_VECTOR_REAL,
  UNDEFINED
};

inline Edt typeInfoToEdt(const std::type_info& tp) {
  if (essentia::sameType(tp, typeid(essentia::Real))) return REAL;
  if (essentia::sameType(tp, typeid(std::string))) return STRING;
  if (essentia::sameType(tp, typeid(int))) return INTEGER;
  if (essentia::sameType(tp, typeid(bool))) return BOOL;
  if (essentia::sameType(tp, typeid(essentia::StereoSample))) return STEREOSAMPLE;
  if (essentia::sameType(tp, typeid(std::vector<essentia::Real>))) return VECTOR_REAL;
  if (essentia::sameType(tp, typeid(std::vector<std::string>))) return VECTOR_STRING;
  if (essentia::sameType(tp, typeid(std::vector<std::complex<essentia::Real> >))) return VECTOR_COMPLEX;
  if (essentia::sameType(tp, typeid(std::vector<int>))) return VECTOR_INTEGER;
  if (essentia::sameType(tp, typeid(std::vector<essentia::StereoSample>))) return VECTOR_STEREOSAMPLE;
  if (essentia::sameType(tp, typeid(std::vector<std::vector<essentia::Real> >))) return VECTOR_VECTOR_REAL;
  if (essentia::sameType(tp, typeid(std::vector<std::vector<std::string> >))) return VECTOR_VECTOR_STRING;
  if (essentia::sameType(tp, typeid(std::vector<std::vector<essentia::StereoSample> >))) return VECTOR_VECTOR_STEREOSAMPLE;
  if (essentia::sameType(tp, typeid(TNT::Array2D<essentia::Real>))) return MATRIX_REAL;
  if (essentia::sameType(tp, typeid(std::vector<TNT::Array2D<essentia::Real> >))) return VECTOR_MATRIX_REAL;
  if (essentia::sameType(tp, typeid(essentia::Pool))) return POOL;
  return UNDEFINED;
}

inline std::string edtToString(Edt tp) {
  switch (tp) {
    case REAL: return "REAL";
    case STRING: return "STRING";
    case INTEGER: return "INTEGER";
    case BOOL: return "BOOL";
    case STEREOSAMPLE: return "STEREOSAMPLE";
    case VECTOR_REAL: return "VECTOR_REAL";
    case VECTOR_STRING: return "VECTOR_STRING";
    case VECTOR_COMPLEX: return "VECTOR_COMPLEX";
    case VECTOR_INTEGER: return "VECTOR_INTEGER";
    case VECTOR_STEREOSAMPLE: return "VECTOR_STEREOSAMPLE";
    case VECTOR_VECTOR_REAL: return "VECTOR_VECTOR_REAL";
    case VECTOR_VECTOR_STRING: return "VECTOR_VECTOR_STRING";
    case VECTOR_VECTOR_STEREOSAMPLE: return "VECTOR_VECTOR_STEREOSAMPLE";
    case MATRIX_REAL: return "MATRIX_REAL";
    case VECTOR_MATRIX_REAL: return "VECTOR_MATRIX_REAL";
    case POOL: return "POOL";
    case MAP_VECTOR_STRING: return "MAP_VECTOR_STRING";
    default: return "UNDEFINED";
  }
}

inline Edt stringToEdt(const std::string& tpName) {
  if (tpName == "REAL") return REAL;
  if (tpName == "STRING") return STRING;
  if (tpName == "INTEGER") return INTEGER;
  if (tpName == "BOOL") return BOOL;
  if (tpName == "STEREOSAMPLE") return STEREOSAMPLE;
  if (tpName == "VECTOR_REAL") return VECTOR_REAL;
  if (tpName == "VECTOR_STRING") return VECTOR_STRING;
  if (tpName == "VECTOR_COMPLEX") return VECTOR_COMPLEX;
  if (tpName == "VECTOR_INTEGER") return VECTOR_INTEGER;
  if (tpName == "VECTOR_STEREOSAMPLE") return VECTOR_STEREOSAMPLE;
  if (tpName == "VECTOR_VECTOR_REAL") return VECTOR_VECTOR_REAL;
  if (tpName == "VECTOR_VECTOR_STRING") return VECTOR_VECTOR_STRING;
  if (tpName == "VECTOR_VECTOR_STEREOSAMPLE") return VECTOR_VECTOR_STEREOSAMPLE;
  if (tpName == "MATRIX_REAL") return MATRIX_REAL;
  if (tpName == "VECTOR_MATRIX_REAL") return VECTOR_MATRIX_REAL;
  if (tpName == "POOL") return POOL;
  if (tpName == "MAP_VECTOR_STRING") return MAP_VECTOR_STRING;
  return UNDEFINED;
}

inline Edt paramTypeToEdt(const essentia::Parameter::ParamType& p) {
  switch (p) {
    case essentia::Parameter::UNDEFINED: return UNDEFINED;
    case essentia::Parameter::STRING: return STRING;
    case essentia::Parameter::REAL: return REAL;
    case essentia::Parameter::BOOL: return BOOL;
    case essentia::Parameter::INT: return INTEGER;
    case essentia::Parameter::STEREOSAMPLE: return STEREOSAMPLE;
    case essentia::Parameter::MATRIX_REAL: return MATRIX_REAL;
    case essentia::Parameter::VECTOR_REAL: return VECTOR_REAL;
    case essentia::Parameter::VECTOR_STRING: return VECTOR_STRING;
    case essentia::Parameter::VECTOR_INT: return VECTOR_INTEGER;
    case essentia::Parameter::VECTOR_STEREOSAMPLE: return VECTOR_STEREOSAMPLE;
    case essentia::Parameter::MAP_VECTOR_REAL: return MAP_VECTOR_REAL;
    case essentia::Parameter::MAP_VECTOR_STRING: return MAP_VECTOR_STRING;

    default:
      std::ostringstream msg;
      msg << "Unable to convert Parameter type to Edt type: " << p;
      throw essentia::EssentiaException(msg.str());
  }
}

inline void* allocate(Edt tp) {
  switch (tp) {
    case REAL: return new essentia::Real;
    case STRING: return new std::string;
    case BOOL: return new bool;
    case INTEGER: return new int;
    case STEREOSAMPLE: return new essentia::StereoSample;
    case VECTOR_REAL: return new essentia::RogueVector<essentia::Real>;
    case VECTOR_STRING: return new std::vector<std::string>;
    case VECTOR_INTEGER: return new essentia::RogueVector<int>;
    case VECTOR_COMPLEX: return new essentia::RogueVector<std::complex<essentia::Real> >;
    case VECTOR_STEREOSAMPLE: return new std::vector<essentia::StereoSample>;
    case VECTOR_VECTOR_REAL: return new std::vector<std::vector<essentia::Real> >;
    case VECTOR_VECTOR_STRING: return new std::vector<std::vector<std::string> >;
    case VECTOR_VECTOR_STEREOSAMPLE: return new std::vector<std::vector<essentia::StereoSample> >;
    case MATRIX_REAL: return new TNT::Array2D<essentia::Real>;
    case VECTOR_MATRIX_REAL: return new std::vector<TNT::Array2D<essentia::Real> >;
    case POOL: return new essentia::Pool;
    default:
      throw essentia::EssentiaException("alloc: allocation of this type is unimplemented: ", edtToString(tp));
  }
}

inline void dealloc(void* ptr, Edt tp) {
  switch (tp) {
    case REAL: delete (essentia::Real*)ptr; break;
    case STRING: delete (std::string*)ptr; break;
    case BOOL: delete (bool*)ptr; break;
    case INTEGER: delete (int*)ptr; break;
    case STEREOSAMPLE: delete (essentia::StereoSample*)ptr; break;
    case VECTOR_REAL: delete (essentia::RogueVector<essentia::Real>*)ptr; break;
    case VECTOR_INTEGER: delete (essentia::RogueVector<int>*)ptr; break;
    case VECTOR_COMPLEX: delete (essentia::RogueVector<std::complex<essentia::Real> >*)ptr; break;
    case VECTOR_STRING: delete (std::vector<std::string>*)ptr; break;
    case VECTOR_STEREOSAMPLE: delete (std::vector<essentia::StereoSample>*)ptr; break;
    case VECTOR_VECTOR_REAL: delete (std::vector<std::vector<essentia::Real> >*)ptr; break;
    case VECTOR_VECTOR_STRING: delete (std::vector<std::vector<std::string> >*)ptr; break;
    case VECTOR_VECTOR_STEREOSAMPLE: delete (std::vector<std::vector<essentia::StereoSample> >*)ptr; break;
    case MATRIX_REAL: delete (TNT::Array2D<essentia::Real>*)ptr; break;
    case VECTOR_MATRIX_REAL: delete (std::vector<TNT::Array2D<essentia::Real> >*)ptr; break;
    case POOL: delete (essentia::Pool*)ptr; break;
    default:
      throw essentia::EssentiaException("dealloc: deallocation of this type is unimplemented: ", edtToString(tp));
  }
}

inline std::string strtype(PyObject* obj) {
  return PyString_AsString(PyObject_Str(PyObject_Type(obj)));
}

DECLARE_PROXY_TYPE(PyReal, essentia::Real);
DECLARE_PYTHON_TYPE(PyReal);

DECLARE_PROXY_TYPE(String, std::string);
DECLARE_PYTHON_TYPE(String);

DECLARE_PROXY_TYPE(Integer, int);
DECLARE_PYTHON_TYPE(Integer);

DECLARE_PROXY_TYPE(Boolean, bool);
DECLARE_PYTHON_TYPE(Boolean);

DECLARE_PROXY_TYPE(PyStereoSample, essentia::StereoSample);
DECLARE_PYTHON_TYPE(PyStereoSample);

DECLARE_PROXY_TYPE(VectorInteger, essentia::RogueVector<int>);
DECLARE_PYTHON_TYPE(VectorInteger);

DECLARE_PROXY_TYPE(VectorReal, essentia::RogueVector<essentia::Real>);
DECLARE_PYTHON_TYPE(VectorReal);

DECLARE_PROXY_TYPE(VectorString, std::vector<std::string>);
DECLARE_PYTHON_TYPE(VectorString);

DECLARE_PROXY_TYPE(VectorComplex, essentia::RogueVector<std::complex<essentia::Real> >);
DECLARE_PYTHON_TYPE(VectorComplex);

DECLARE_PROXY_TYPE(VectorStereoSample, std::vector<essentia::StereoSample>);
DECLARE_PYTHON_TYPE(VectorStereoSample);

DECLARE_PROXY_TYPE(VectorVectorReal, std::vector<std::vector<essentia::Real> >);
DECLARE_PYTHON_TYPE(VectorVectorReal);

DECLARE_PROXY_TYPE(VectorVectorString, std::vector<std::vector<std::string> >);
DECLARE_PYTHON_TYPE(VectorVectorString);

DECLARE_PROXY_TYPE(VectorVectorStereoSample, std::vector<std::vector<essentia::StereoSample> >);
DECLARE_PYTHON_TYPE(VectorVectorStereoSample);

DECLARE_PROXY_TYPE(MatrixReal, TNT::Array2D<essentia::Real>);
DECLARE_PYTHON_TYPE(MatrixReal);

DECLARE_PROXY_TYPE(VectorMatrixReal, std::vector<TNT::Array2D<essentia::Real> >);
DECLARE_PYTHON_TYPE(VectorMatrixReal);

// need to use a typedef here because of the macro usage
typedef std::map<std::string, std::vector<std::string> > mapvectorstring;

DECLARE_PROXY_TYPE(MapVectorString, mapvectorstring);
DECLARE_PYTHON_TYPE(MapVectorString);


#endif // ESSENTIA_PYTHON_TYPEDEFS_H
