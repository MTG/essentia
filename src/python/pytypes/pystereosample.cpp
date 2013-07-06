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
#include "parsing.h"
using namespace std;
using namespace essentia;

DEFINE_PYTHON_TYPE(PyStereoSample);

PyObject* PyStereoSample::toPythonCopy(const StereoSample* x) {
  PyObject* pyx = PyTuple_Pack(2, PyFloat_FromDouble(x->left()),
                                  PyFloat_FromDouble(x->right()));

  if (pyx == NULL) {
    throw EssentiaException("PyStereoSample::toPythonCopy: could not create tuple");
  }

  return pyx;
}


void* PyStereoSample::fromPythonCopy(PyObject* obj) {
  if (!PyTuple_Check(obj)) {
    throw EssentiaException("PyStereoSample::fromPythonCopy: input not a tuple: ", strtype(obj));
  }

  if (PyTuple_GET_SIZE(obj) != 2) {
    throw EssentiaException("PyStereoSample::fromPythonCopy: input tuple is not of size 2: ", PyTuple_GET_SIZE(obj));
  }

  // extract stereo values
  Real* left = reinterpret_cast<Real*>(PyReal::fromPythonCopy( PyTuple_GET_ITEM(obj, 0) ));
  Real* right = reinterpret_cast<Real*>(PyReal::fromPythonCopy( PyTuple_GET_ITEM(obj, 1) ));

  StereoSample* ss = new StereoSample();
  ss->left() = *left;
  ss->right() = *right;

  delete left;
  delete right;

  return ss;
}
