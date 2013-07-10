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

DEFINE_PYTHON_TYPE(VectorInteger);

PyObject* VectorInteger::toPythonRef(RogueVector<int>* v) {
  npy_intp dim = v->size();
  PyObject* result;

  if (dim > 0) result = PyArray_SimpleNewFromData(1, &dim, PyArray_INT, &((*v)[0]));
  else         result = PyArray_SimpleNew(1, &dim, PyArray_INT);

  if (result == NULL) {
    throw EssentiaException("VectorInteger::toPythonRef: could not create PyArray of type PyArray_INT");
  }

  PyArray_BASE(result) = TO_PYTHON_PROXY(VectorInteger, v);

  return result;
}


void* VectorInteger::fromPythonRef(PyObject* obj) {
  // if input is a numpy array, just wrap its data with the RogueVector
  if (!PyArray_Check(obj)) {
    throw EssentiaException("VectorInteger::fromPythonRef: input is not a PyArray");
  }

  PyArrayObject* array = (PyArrayObject*)obj;

  if (array->descr->type_num != PyArray_INT32) {
    throw EssentiaException("VectorInteger::fromPythonRef: this NumPy array doesn't contain ints (maybe you forgot dtype='int'), type code: ", array->descr->type_num);
  }
  if (array->nd != 1) {
    throw EssentiaException("VectorInteger::fromPythonRef: this NumPy array has dimension ", array->nd, " (expected 1)");
  }

  return new RogueVector<int>((int*)PyArray_DATA(obj), PyArray_SIZE(obj));
}

Parameter* VectorInteger::toParameter(PyObject* obj) {
  RogueVector<int>* value = (RogueVector<int>*)fromPythonRef(obj);
  Parameter* result = new Parameter(*value);
  delete value;
  return result;
}
