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

DEFINE_PYTHON_TYPE(VectorVectorStereoSample);

PyObject* VectorVectorStereoSample::toPythonCopy(const vector<vector<StereoSample> >* v) {
  npy_intp dims[2] = { 0, 0 };
  dims[0] = v->size();
  if (!v->empty()) dims[1] = (*v)[0].size();

  bool isRectangular = true;

  // check all rows have the same size
  for (int i=1; i<dims[0]; i++) {
    if ((int)(*v)[i].size() != dims[1]) isRectangular = false;
  }

  if (isRectangular) {
    PyArrayObject* result = (PyArrayObject*)PyArray_SimpleNew(2, dims, PyArray_OBJECT);

    if (result == NULL) {
      throw EssentiaException("VectorVectorStereoSample: dang null object");
      cout << "dang null object" << endl;
    }

    for (int i=0; i<dims[0]; i++) {
      for (int j=0; j<dims[1]; j++) {
        PyObject** ptr = (PyObject**)(result->data + i*result->strides[0] + j*result->strides[1]);
        *ptr = PyStereoSample::toPythonCopy(&((*v)[i][j]));
      }
    }

    return (PyObject*)result;
  }
  else {
    PyObject* result = PyList_New(v->size());

    for (int i=0; i<(int)v->size(); i++) {
      PyObject* item = PyList_New((*v)[i].size());

      for (int j=0; j<(int)(*v)[i].size(); j++) {
        PyList_SET_ITEM(item, j, PyStereoSample::toPythonCopy(&((*v)[i][j])));
      }

      PyList_SET_ITEM(result, i, item);
    }

    return result;
  }
}

void* VectorVectorStereoSample::fromPythonRef(PyObject* obj) {
  // if input is a numpy array, just wrap its data with the RogueVector
  if (PyArray_Check(obj)) {
    // not implemented atm, would need to make a RogueVector<RogueVector<Real> >...
    throw EssentiaException("Not implement at the moment");
  }

  if (PyList_Check(obj)) {
    int size = PyList_Size(obj);
    vector<vector<StereoSample> >* v = new vector<vector<StereoSample> >(size, vector<StereoSample>());

    for (int i=0; i<size; i++) {
      PyObject* row = PyList_GetItem(obj, i);
      if (!PyList_Check(obj)) {
        cout << "VectorVectorStereoSample: Not all elements in the list are of the same type: "
             << "element " << i << " has type " << strtype(row) << endl;
        delete v;
        return NULL;
      }

      int rowsize = PyList_Size(row);
      (*v)[i].resize(rowsize);

      for (int j=0; j<rowsize; j++) {
        PyObject* item = PyList_GetItem(row, j);
          PyStereoSample* sample = reinterpret_cast<PyStereoSample*>(PyStereoSample::fromPythonCopy(PyList_GetItem(obj, i)));
        if (sample == NULL) {
          cout << "VectorVectorStereoSample::fromPythonRef: not all elements in the matrix are of the same type: " << "element (" << i << ", " << j <<") has type " <<strtype(item) << endl;
          delete v;
          return NULL;
        }
        (*v)[i][j] = *(sample->data);
      }
    }

    return v;
  }

  cout << "VectorVectorStereoSample::fromPython: not a vector<StereoSample>: " << strtype(obj) << endl;
  return NULL;
}
