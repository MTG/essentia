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

DEFINE_PYTHON_TYPE(String);

PyObject* String::toPythonCopy(const string* s) {
  return PyString_FromStringAndSize(s->c_str(), s->size());
}


void* String::fromPythonCopy(PyObject* obj) {
  if (PyString_Check(obj)) {
    return new string(PyString_AS_STRING(obj));
  }

  if (PyUnicode_Check(obj)) {
    PyObject* utf8str = PyUnicode_AsEncodedString(obj, "utf-8", 0);
    if (!utf8str) {
      E_ERROR("Error converting unicode to utf8 string");
      return new string("");
    }

    string* result = new string(PyString_AS_STRING(utf8str));
    Py_DECREF(utf8str);
    return result;
  }

  throw EssentiaException("String::fromPythonCopy: input not a PyString: ", strtype(obj));
}

Parameter* String::toParameter(PyObject* obj) {
  string* value = (string*)fromPythonCopy(obj);
  Parameter* result = new Parameter(*value);
  delete value;
  return result;
}
