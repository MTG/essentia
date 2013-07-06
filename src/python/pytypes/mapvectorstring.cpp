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


DEFINE_PYTHON_TYPE(MapVectorString);


PyObject* MapVectorString::toPythonCopy(const map<string, vector<string> >* v) {
  throw EssentiaException("MapVectorString::fromPythonCopy currently not implemented");
}


void* MapVectorString::fromPythonCopy(PyObject* obj) {
  if (!PyDict_Check(obj)) {
    throw EssentiaException("MapVectorString::fromPythonCopy: expected PyDict, instead received: ", strtype(obj));
  }

  throw EssentiaException("MapVectorString::fromPythonCopy currently not implemented");
}

Parameter* MapVectorString::toParameter(PyObject* obj) {
  map<string, vector<string> >* value = (map<string, vector<string> >*)fromPythonCopy(obj);
  Parameter* result = new Parameter(*value);
  delete value;
  return result;
}
