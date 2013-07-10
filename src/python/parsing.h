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

#ifndef ESSENTIA_PYTHON_PARSING_H
#define ESSENTIA_PYTHON_PARSING_H

#include <Python.h>
#include "parameter.h"
#include "typedefs.h"


#define PARSE_OK 1
#define PARSE_FAILED 0


/**
  @param params represents the default parameters for an algorithm and will have
                the new parsed parameters placed in it
 */
void parseParameters(essentia::ParameterMap* params, PyObject* args, PyObject* keywds);


/**
 * This function unpacks a python tuple into a vector of separate PyObjects.
 */
std::vector<PyObject*> unpack(PyObject* args);


/**
 * This function builds the python object that is to be returned given
 * a vector of outputs. It automatically chooses the correct output type
 * (None, simple value, tuple of values) wrt the number of outputs.
 */
PyObject* buildReturnValue(const std::vector<PyObject*>& result_vec);

PyObject* toPython(void* obj, Edt tp);

PyObject* paramToPython(const essentia::Parameter& p);

#endif // ESSENTIA_PYTHON_PARSING_H
