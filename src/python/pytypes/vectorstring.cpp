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

#include "typedefs.h"
using namespace std;
using namespace essentia;


DEFINE_PYTHON_TYPE(VectorString);


PyObject* VectorString::toPythonCopy(const vector<string>* v) {
  int size = v->size();
  PyObject* result = PyList_New(size);

  for (int i=0; i<size; ++i) {
    PyList_SET_ITEM(result, i, PyString_FromString((*v)[i].c_str()));
  }

  return result;
}


void* VectorString::fromPythonCopy(PyObject* obj) {
  // if input is a list of strings, creates a copy vector
  if (!PyList_Check(obj)) {
    throw EssentiaException("VectorString::fromPythonCopy: expected PyList, instead received: ", strtype(obj));
  }

  int size = int(PyList_Size(obj));
  vector<string>* v = new vector<string>(size, "");

  for (int i=0; i<size; ++i) {
    PyObject* item = PyList_GET_ITEM(obj, i);
    if (!PyString_Check(item)) {
      delete v;
      throw EssentiaException("VectorString::fromPythonCopy: all elements of PyList must be strings, found: ", strtype(item));
    }
    (*v)[i] = PyString_AS_STRING(item);
  }

  return v;
}

Parameter* VectorString::toParameter(PyObject* obj) {
  vector<string>* value = (vector<string>*)fromPythonCopy(obj);
  Parameter* result = new Parameter(*value);
  delete value;
  return result;
}
