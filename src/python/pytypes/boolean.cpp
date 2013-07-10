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

DEFINE_PYTHON_TYPE(Boolean);

PyObject* Boolean::toPythonCopy(const bool* x) {
  if (*x) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}


void* Boolean::fromPythonCopy(PyObject* obj) {
  if (!PyBool_Check(obj)) {
    throw EssentiaException("Boolean::fromPythonCopy: input is not a PyBool");
  }

  return new bool(obj == Py_True);
}

Parameter* Boolean::toParameter(PyObject* obj) {
  bool* value = (bool*)fromPythonCopy(obj);
  Parameter* result = new Parameter(*value);
  delete value;
  return result;
}
