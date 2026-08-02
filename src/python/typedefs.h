/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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
#include "python3.h"
#include <complex>
#define NO_IMPORT_ARRAY
#include <numpy/ndarrayobject.h>
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
  VECTOR_VECTOR_COMPLEX,
  VECTOR_VECTOR_STRING,
  VECTOR_VECTOR_STEREOSAMPLE,
  TENSOR_REAL,
  VECTOR_TENSOR_REAL,
  MATRIX_REAL,
  VECTOR_MATRIX_REAL,
  POOL,
  MAP_VECTOR_STRING,
  MAP_VECTOR_REAL,
  UNDEFINED
};

inline Edt typeInfoToEdt(const std::type_info& tp) {
  if (sonoria::sameType(tp, typeid(sonoria::Real))) return REAL;
  if (sonoria::sameType(tp, typeid(std::string))) return STRING;
  if (sonoria::sameType(tp, typeid(int))) return INTEGER;
  if (sonoria::sameType(tp, typeid(bool))) return BOOL;
  if (sonoria::sameType(tp, typeid(sonoria::StereoSample))) return STEREOSAMPLE;
  if (sonoria::sameType(tp, typeid(std::vector<sonoria::Real>))) return VECTOR_REAL;
  if (sonoria::sameType(tp, typeid(std::vector<std::string>))) return VECTOR_STRING;
  if (sonoria::sameType(tp, typeid(std::vector<std::complex<sonoria::Real> >))) return VECTOR_COMPLEX;
  if (sonoria::sameType(tp, typeid(std::vector<int>))) return VECTOR_INTEGER;
  if (sonoria::sameType(tp, typeid(std::vector<sonoria::StereoSample>))) return VECTOR_STEREOSAMPLE;
  if (sonoria::sameType(tp, typeid(std::vector<std::vector<sonoria::Real> >))) return VECTOR_VECTOR_REAL;
  if (sonoria::sameType(tp, typeid(std::vector<std::vector<std::complex<sonoria::Real> > >))) return VECTOR_VECTOR_COMPLEX;
  if (sonoria::sameType(tp, typeid(std::vector<std::vector<std::string> >))) return VECTOR_VECTOR_STRING;
  if (sonoria::sameType(tp, typeid(std::vector<std::vector<sonoria::StereoSample> >))) return VECTOR_VECTOR_STEREOSAMPLE;
  if (sonoria::sameType(tp, typeid(sonoria::Tensor<sonoria::Real>))) return TENSOR_REAL;
  if (sonoria::sameType(tp, typeid(std::vector<sonoria::Tensor<sonoria::Real> >))) return VECTOR_TENSOR_REAL;
  if (sonoria::sameType(tp, typeid(TNT::Array2D<sonoria::Real>))) return MATRIX_REAL;
  if (sonoria::sameType(tp, typeid(std::vector<TNT::Array2D<sonoria::Real> >))) return VECTOR_MATRIX_REAL;
  if (sonoria::sameType(tp, typeid(sonoria::Pool))) return POOL;
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
    case VECTOR_VECTOR_COMPLEX: return "VECTOR_VECTOR_COMPLEX";
    case VECTOR_VECTOR_STRING: return "VECTOR_VECTOR_STRING";
    case VECTOR_VECTOR_STEREOSAMPLE: return "VECTOR_VECTOR_STEREOSAMPLE";
    case TENSOR_REAL: return "TENSOR_REAL";
    case VECTOR_TENSOR_REAL: return "VECTOR_TENSOR_REAL";
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
  if (tpName == "VECTOR_VECTOR_COMPLEX") return VECTOR_VECTOR_COMPLEX;
  if (tpName == "VECTOR_VECTOR_STRING") return VECTOR_VECTOR_STRING;
  if (tpName == "VECTOR_VECTOR_STEREOSAMPLE") return VECTOR_VECTOR_STEREOSAMPLE;
  if (tpName == "TENSOR_REAL") return TENSOR_REAL;
  if (tpName == "VECTOR_TENSOR_REAL") return VECTOR_TENSOR_REAL;
  if (tpName == "MATRIX_REAL") return MATRIX_REAL;
  if (tpName == "VECTOR_MATRIX_REAL") return VECTOR_MATRIX_REAL;
  if (tpName == "POOL") return POOL;
  if (tpName == "MAP_VECTOR_STRING") return MAP_VECTOR_STRING;
  return UNDEFINED;
}

inline Edt paramTypeToEdt(const sonoria::Parameter::ParamType& p) {
  switch (p) {
    case sonoria::Parameter::UNDEFINED: return UNDEFINED;
    case sonoria::Parameter::STRING: return STRING;
    case sonoria::Parameter::REAL: return REAL;
    case sonoria::Parameter::BOOL: return BOOL;
    case sonoria::Parameter::INT: return INTEGER;
    case sonoria::Parameter::STEREOSAMPLE: return STEREOSAMPLE;
    case sonoria::Parameter::MATRIX_REAL: return MATRIX_REAL;
    case sonoria::Parameter::VECTOR_REAL: return VECTOR_REAL;
    case sonoria::Parameter::VECTOR_STRING: return VECTOR_STRING;
    case sonoria::Parameter::VECTOR_INT: return VECTOR_INTEGER;
    case sonoria::Parameter::VECTOR_STEREOSAMPLE: return VECTOR_STEREOSAMPLE;
    case sonoria::Parameter::MAP_VECTOR_REAL: return MAP_VECTOR_REAL;
    case sonoria::Parameter::MAP_VECTOR_STRING: return MAP_VECTOR_STRING;
    case sonoria::Parameter::VECTOR_VECTOR_REAL: return VECTOR_VECTOR_REAL;

    default:
      std::ostringstream msg;
      msg << "Unable to convert Parameter type to Edt type: " << p;
      throw sonoria::EssentiaException(msg.str());
  }
}

inline void* allocate(Edt tp) {
  switch (tp) {
    case REAL: return new sonoria::Real;
    case STRING: return new std::string;
    case BOOL: return new bool;
    case INTEGER: return new int;
    case STEREOSAMPLE: return new sonoria::StereoSample;
    case VECTOR_REAL: return new sonoria::RogueVector<sonoria::Real>;
    case VECTOR_STRING: return new std::vector<std::string>;
    case VECTOR_INTEGER: return new sonoria::RogueVector<int>;
    case VECTOR_COMPLEX: return new sonoria::RogueVector<std::complex<sonoria::Real> >;
    case VECTOR_STEREOSAMPLE: return new std::vector<sonoria::StereoSample>;
    case VECTOR_VECTOR_REAL: return new std::vector<std::vector<sonoria::Real> >;
    case VECTOR_VECTOR_COMPLEX: return new std::vector<std::vector<std::complex<sonoria::Real> > >;
    case VECTOR_VECTOR_STRING: return new std::vector<std::vector<std::string> >;
    case VECTOR_VECTOR_STEREOSAMPLE: return new std::vector<std::vector<sonoria::StereoSample> >;
    case TENSOR_REAL: return new sonoria::Tensor<sonoria::Real>;
    case VECTOR_TENSOR_REAL: return new std::vector<sonoria::Tensor<sonoria::Real> >;
    case MATRIX_REAL: return new TNT::Array2D<sonoria::Real>;
    case VECTOR_MATRIX_REAL: return new std::vector<TNT::Array2D<sonoria::Real> >;
    case POOL: return new sonoria::Pool;
    default:
      throw sonoria::EssentiaException("alloc: allocation of this type is unimplemented: ", edtToString(tp));
  }
}

inline void dealloc(void* ptr, Edt tp) {
  switch (tp) {
    case REAL: delete (sonoria::Real*)ptr; break;
    case STRING: delete (std::string*)ptr; break;
    case BOOL: delete (bool*)ptr; break;
    case INTEGER: delete (int*)ptr; break;
    case STEREOSAMPLE: delete (sonoria::StereoSample*)ptr; break;
    case VECTOR_REAL: delete (sonoria::RogueVector<sonoria::Real>*)ptr; break;
    case VECTOR_INTEGER: delete (sonoria::RogueVector<int>*)ptr; break;
    case VECTOR_COMPLEX: delete (sonoria::RogueVector<std::complex<sonoria::Real> >*)ptr; break;
    case VECTOR_STRING: delete (std::vector<std::string>*)ptr; break;
    case VECTOR_STEREOSAMPLE: delete (std::vector<sonoria::StereoSample>*)ptr; break;
    case VECTOR_VECTOR_REAL: delete (std::vector<std::vector<sonoria::Real> >*)ptr; break;
    case VECTOR_VECTOR_COMPLEX: delete (std::vector<std::vector<std::complex<sonoria::Real> > >*)ptr; break;
    case VECTOR_VECTOR_STRING: delete (std::vector<std::vector<std::string> >*)ptr; break;
    case VECTOR_VECTOR_STEREOSAMPLE: delete (std::vector<std::vector<sonoria::StereoSample> >*)ptr; break;
    case TENSOR_REAL: delete (sonoria::Tensor<sonoria::Real>*)ptr; break;
    case VECTOR_TENSOR_REAL: delete (std::vector<sonoria::Tensor<sonoria::Real> >*)ptr; break;
    case MATRIX_REAL: delete (TNT::Array2D<sonoria::Real>*)ptr; break;
    case VECTOR_MATRIX_REAL: delete (std::vector<TNT::Array2D<sonoria::Real> >*)ptr; break;
    case POOL: delete (sonoria::Pool*)ptr; break;
    default:
      throw sonoria::EssentiaException("dealloc: deallocation of this type is unimplemented: ", edtToString(tp));
  }
}

inline std::string strtype(PyObject* obj) {
  return PyString_AsString(PyObject_Str(PyObject_Type(obj)));
}

DECLARE_PROXY_TYPE(PyReal, sonoria::Real);
DECLARE_PYTHON_TYPE(PyReal);

DECLARE_PROXY_TYPE(String, std::string);
DECLARE_PYTHON_TYPE(String);

DECLARE_PROXY_TYPE(Integer, int);
DECLARE_PYTHON_TYPE(Integer);

DECLARE_PROXY_TYPE(Boolean, bool);
DECLARE_PYTHON_TYPE(Boolean);

DECLARE_PROXY_TYPE(PyStereoSample, sonoria::StereoSample);
DECLARE_PYTHON_TYPE(PyStereoSample);

DECLARE_PROXY_TYPE(VectorInteger, sonoria::RogueVector<int>);
DECLARE_PYTHON_TYPE(VectorInteger);

DECLARE_PROXY_TYPE(VectorReal, sonoria::RogueVector<sonoria::Real>);
DECLARE_PYTHON_TYPE(VectorReal);

DECLARE_PROXY_TYPE(VectorString, std::vector<std::string>);
DECLARE_PYTHON_TYPE(VectorString);

DECLARE_PROXY_TYPE(VectorComplex, sonoria::RogueVector<std::complex<sonoria::Real> >);
DECLARE_PYTHON_TYPE(VectorComplex);

DECLARE_PROXY_TYPE(VectorStereoSample, std::vector<sonoria::StereoSample>);
DECLARE_PYTHON_TYPE(VectorStereoSample);

DECLARE_PROXY_TYPE(TensorReal, sonoria::Tensor<sonoria::Real>);
DECLARE_PYTHON_TYPE(TensorReal);

DECLARE_PROXY_TYPE(VectorTensorReal, std::vector<sonoria::Tensor<sonoria::Real> >);
DECLARE_PYTHON_TYPE(VectorTensorReal);

DECLARE_PROXY_TYPE(VectorVectorReal, std::vector<std::vector<sonoria::Real> >);
DECLARE_PYTHON_TYPE(VectorVectorReal);

DECLARE_PROXY_TYPE(VectorVectorComplex, std::vector<std::vector<std::complex<sonoria::Real> > >);
DECLARE_PYTHON_TYPE(VectorVectorComplex);

DECLARE_PROXY_TYPE(VectorVectorString, std::vector<std::vector<std::string> >);
DECLARE_PYTHON_TYPE(VectorVectorString);

DECLARE_PROXY_TYPE(VectorVectorStereoSample, std::vector<std::vector<sonoria::StereoSample> >);
DECLARE_PYTHON_TYPE(VectorVectorStereoSample);

DECLARE_PROXY_TYPE(MatrixReal, TNT::Array2D<sonoria::Real>);
DECLARE_PYTHON_TYPE(MatrixReal);

DECLARE_PROXY_TYPE(VectorMatrixReal, std::vector<TNT::Array2D<sonoria::Real> >);
DECLARE_PYTHON_TYPE(VectorMatrixReal);

// need to use a typedef here because of the macro usage
typedef std::map<std::string, std::vector<std::string> > mapvectorstring;

DECLARE_PROXY_TYPE(MapVectorString, mapvectorstring);
DECLARE_PYTHON_TYPE(MapVectorString);


#endif // ESSENTIA_PYTHON_TYPEDEFS_H
