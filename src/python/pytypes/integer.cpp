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

DEFINE_PYTHON_TYPE(Integer);

PyObject* Integer::toPythonCopy(const int* x) {
  return PyInt_FromLong(*x);
}


void* Integer::fromPythonCopy(PyObject* obj) {
  if (!PyInt_Check(obj)) {
    throw EssentiaException("Integer::fromPythonCopy: input is not a PyInt");
  }

  return new int(PyInt_AsLong(obj));
}

Parameter* Integer::toParameter(PyObject* obj) {
  int* value = (int*)fromPythonCopy(obj);
  Parameter* result = new Parameter(*value);
  delete value;
  return result;
}
