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

#include <Python.h>
#include "parsing.h"
#include "typedefs.h"
using namespace std;
using namespace essentia;


Parameter* PythonDictToParameter(PyObject* dict, Edt tp) {
  if (!PyDict_Check(dict)) throw EssentiaException("map parameter not a python dictionary");

  PyObject *key, *value;
  Py_ssize_t pos = 0;

  switch (tp) {
    case MAP_VECTOR_REAL: {
      map<string, vector<Real> > mapVecReal;
      while (PyDict_Next(dict, &pos, &key, &value)) {
        if (!PyString_Check(key)) throw EssentiaException("all keys in the dict should be strings");

        string skey = PyString_AS_STRING(key);
        RogueVector<Real>* rv = (RogueVector<Real>*)VectorReal::fromPythonRef(value);
        mapVecReal[skey] = *((vector<Real>*)rv);
        delete rv;
      }
      return new Parameter(mapVecReal);
    }

    case MAP_VECTOR_STRING: {
      map<string, vector<string> > mapVecString;
      while (PyDict_Next(dict, &pos, &key, &value)) {
        if (!PyString_Check(key)) throw EssentiaException("all keys in the dict should be strings");

        string skey = PyString_AS_STRING(key);
        mapVecString[skey] = *((vector<string>*)VectorString::fromPythonCopy(value));
      }
      return new Parameter(mapVecString);
    }

    default:
      throw EssentiaException("map type not supported");
  }
}

void parseParameters(ParameterMap* pm, PyObject* args, PyObject* keywds) {
  Py_ssize_t pos = 0;
  PyObject* pyname;
  PyObject* obj;

  while (PyDict_Next(keywds, &pos, &pyname, &obj)) {
    string name = PyString_AsString(pyname);
    Edt tp = paramTypeToEdt((*pm)[name].type());

    switch (tp) {
      case BOOL:              pm->add(name, Boolean::toParameter(obj)); break;
      case INTEGER:           pm->add(name, Integer::toParameter(obj)); break;
      case REAL:              pm->add(name, PyReal::toParameter(obj)); break;
      case STRING:            pm->add(name, String::toParameter(obj)); break;
      case VECTOR_STRING:     pm->add(name, VectorString::toParameter(obj)); break;
      case MAP_VECTOR_STRING:
      case MAP_VECTOR_REAL:
      {
        Parameter* p = PythonDictToParameter(obj, tp);
        pm->add(name, *p);
        delete p;
        break;
      }
      case VECTOR_REAL:       pm->add(name, VectorReal::toParameter(obj)); break;
      case VECTOR_INTEGER:    pm->add(name, VectorInteger::toParameter(obj)); break;

      default:
        ostringstream msg;
        msg << "Parsing unsupported parameter type: " << edtToString(tp) << " for '" << name << "'";
        throw EssentiaException(msg.str());
    }
  }
}


/**
 * This function parses a python tuple into a vector of separate PyObjects.
 */
vector<PyObject*> unpack(PyObject* args) {
  if (!PyTuple_Check(args)) {
    throw EssentiaException("Trying to unwrap an object which is not a tuple: "+strtype(args));
  }

  vector<PyObject*> result;
  int ninputs = PyTuple_GET_SIZE(args);
  result.resize(ninputs);

  for (int i=0; i<ninputs; i++) {
    result[i] = PyTuple_GET_ITEM(args, i);
  }

  return result;
}



///**
// * This function builds the python object that is to be returned given
// * a vector of outputs. It automatically chooses the correct output type
// * (None, simple value, tuple of values) wrt the number of outputs.
// */
PyObject* buildReturnValue(const vector<PyObject*>& result_vec) {
  int size = result_vec.size();

  // if there is no result to be returned, return None to the python interpreter
  if (size == 0) Py_RETURN_NONE;

  // if there is only one value, return it directly instead of a 1-sized tuple
  if (size == 1) return result_vec[0];

  // otherwise, create a tuple and fill it with the results
  PyObject* result = PyTuple_New(result_vec.size());

  for (int i=0; i<size; i++) {
    PyTuple_SET_ITEM(result, i, result_vec[i]);
  }

  return result;
}

PyObject* toPython(void* obj, Edt tp) {
  switch (tp) {
    case REAL: return PyReal::toPythonCopy((Real*)obj);
    case STRING: return String::toPythonCopy((string*)obj);
    case INTEGER: return Integer::toPythonCopy((int*)obj);
    case BOOL: return Boolean::toPythonCopy((bool*)obj);
    case STEREOSAMPLE: return PyStereoSample::toPythonCopy((StereoSample*)obj);
    case VECTOR_REAL: return VectorReal::toPythonRef((RogueVector<Real>*)obj);
    case VECTOR_STRING: return VectorString::toPythonCopy((vector<string>*)obj);
    case VECTOR_COMPLEX: return VectorComplex::toPythonRef((RogueVector<complex<Real> >*)obj);
    case VECTOR_INTEGER: return VectorInteger::toPythonRef((RogueVector<int>*)obj);
    case VECTOR_STEREOSAMPLE: return VectorStereoSample::toPythonCopy((vector<StereoSample>*)obj);
    case VECTOR_VECTOR_REAL: return VectorVectorReal::toPythonCopy((vector<vector<Real> >*)obj);
    case VECTOR_VECTOR_STRING: return VectorVectorString::toPythonCopy((vector<vector<string> >*)obj);
    case VECTOR_VECTOR_STEREOSAMPLE: return VectorVectorStereoSample::toPythonCopy((vector<vector<StereoSample> >*)obj);
    case MATRIX_REAL: return MatrixReal::toPythonRef((TNT::Array2D<Real>*)obj);
    case VECTOR_MATRIX_REAL: return VectorMatrixReal::toPythonCopy((vector<TNT::Array2D<Real> >*)obj);
    case POOL: return PyPool::toPythonRef((Pool*)obj);

    // WARNING: This list is very incomplete. For example, paramToPython uses this function and is
    // supposed to support every kind of parameter including Maps of things which are obviously not
    // currently supported in this function. To complete this list, make sure to see paramToPython.

    default:
      throw EssentiaException("toPython: Unable to convert data type to a python type: ", edtToString(tp));
  }
}

typedef map<string, Real> mapreal;
typedef map<string, vector<Real> > mapvectorreal;
typedef map<string, vector<string> > mapvectorstring;
typedef map<string, vector<int> > mapvectorint;

PyObject* paramToPython(const Parameter& p) {
  if (!p.isConfigured()) return NULL;

  Parameter::ParamType pType = p.type();

  #define PARAM_CASE(e, t, fname) case Parameter::e: { t temp = p.to##fname(); return toPython(&temp, paramTypeToEdt(pType)); }


  switch (pType) {
    PARAM_CASE(REAL, Real, Real);
    PARAM_CASE(STRING, string, String);
    PARAM_CASE(INT, int, Int);
    PARAM_CASE(BOOL, bool, Bool);

    PARAM_CASE(STEREOSAMPLE, StereoSample, StereoSample);
    case Parameter::VECTOR_REAL: {
      // we have to do the vectors in a special way since they need to not be deleted (i.e. toPython
      // won't make a copy of them: toPythonRef)
      vector<Real> v = p.toVectorReal();
      RogueVector<Real>* r = new RogueVector<Real>(v.size(), 0);
      for (int i=0; i<int(v.size()); ++i) (*r)[i] = v[i];
      return toPython(r, paramTypeToEdt(pType));
    }
    PARAM_CASE(VECTOR_STRING, vector<string>, VectorString);
    PARAM_CASE(VECTOR_BOOL, vector<bool>, VectorBool);
    case Parameter::VECTOR_INT: {
      vector<int> v = p.toVectorInt();
      RogueVector<int>* r = new RogueVector<int>(v.size(), 0);
      for (int i=0; i<int(v.size()); ++i) (*r)[i] = v[i];
      return toPython(r, paramTypeToEdt(pType));
    }
    PARAM_CASE(VECTOR_STEREOSAMPLE, vector<StereoSample>, VectorStereoSample);
    PARAM_CASE(VECTOR_VECTOR_REAL, vector<vector<Real> >, VectorVectorReal);
    PARAM_CASE(VECTOR_VECTOR_STRING, vector<vector<string> >, VectorVectorString);
    PARAM_CASE(VECTOR_VECTOR_STEREOSAMPLE, vector<vector<StereoSample> >, VectorVectorStereoSample);
    PARAM_CASE(MATRIX_REAL, TNT::Array2D<Real>, MatrixReal);
    PARAM_CASE(VECTOR_MATRIX_REAL, vector<TNT::Array2D<Real> >, VectorMatrixReal);
    PARAM_CASE(MAP_VECTOR_REAL, mapvectorreal, MapVectorReal);
    PARAM_CASE(MAP_VECTOR_STRING, mapvectorstring, MapVectorString);
    PARAM_CASE(MAP_VECTOR_INT, mapvectorint, MapVectorInt);
    PARAM_CASE(MAP_REAL, mapreal, MapReal);

    default: return NULL;
  }

  #undef PARAM_CASE
}
