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
