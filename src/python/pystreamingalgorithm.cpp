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

#include <Python.h>
#include "algorithmfactory.h"
#include "parsing.h"
#include "network.h"
#include "pytypes/pypool.h" // to use its type-determining capabilities
#include "commonfunctions.h"
using namespace std;
using namespace essentia;
using namespace streaming;


#define PY_ALGONAME "Streaming: " << self->algo->name()


class PyStreamingAlgorithm {
 public:
  PyObject_HEAD

  // an algorithm is considered a generator until another algorithm is
  // connected to one of the generator's inputs
  bool isGenerator;
  streaming::Algorithm* algo;

  static PyObject* tp_new(PyTypeObject* subtype, PyObject* args, PyObject* kwds);
  static int tp_init(PyStreamingAlgorithm *self, PyObject *args, PyObject *kwds);
  static void tp_dealloc(PyObject* self);

  static PyObject* name (PyStreamingAlgorithm* self) {
    return toPython((void*)&self->algo->name(), STRING);
  }
  static PyObject* configure (PyStreamingAlgorithm* self, PyObject* args, PyObject* keywds);
  static PyObject* hasSink(PyStreamingAlgorithm* self, PyObject* args);
  static PyObject* hasSource(PyStreamingAlgorithm* self, PyObject* args);
  static PyObject* push(PyStreamingAlgorithm* self, PyObject* obj);

  static PyObject* inputNames(PyStreamingAlgorithm* self) {
    vector<string> names = self->algo->inputNames();
    return toPython((void*)&names, VECTOR_STRING);
  }

  static PyObject* outputNames(PyStreamingAlgorithm* self) {
    vector<string> names = self->algo->outputNames();
    return toPython((void*)&names, VECTOR_STRING);
  }

  static PyObject* parameterNames(PyStreamingAlgorithm* self) {
    vector<string> names = self->algo->defaultParameters().keys();
    return toPython((void*)&names, VECTOR_STRING);
  }

  static PyObject* getInputType(PyStreamingAlgorithm* self, PyObject* obj);
  static PyObject* getOutputType(PyStreamingAlgorithm* self, PyObject* obj);
  static PyObject* paramType(PyStreamingAlgorithm* self, PyObject* name);
  static PyObject* paramValue(PyStreamingAlgorithm* self, PyObject* name);

  static PyObject* getDoc(PyStreamingAlgorithm* self);
  static PyObject* getStruct(PyStreamingAlgorithm* self);
};

PyObject* PyStreamingAlgorithm::tp_new(PyTypeObject* subtype, PyObject* args, PyObject* kwds) {
  return (PyObject*)(subtype->tp_alloc(subtype, 0));
}

int PyStreamingAlgorithm::tp_init(PyStreamingAlgorithm *self, PyObject *args, PyObject *kwds) {
  static char *kwlist[] = { (char*)"name", NULL };
  char* algoname;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &algoname)) {
    return -1;
  }

  try {
    self->algo = streaming::AlgorithmFactory::create(algoname);
    self->isGenerator = true;
  }
  catch (exception& e) {
    ostringstream msg;
    msg << "PyStreamingAlgorithm.init: " << e.what();
    PyErr_SetString(PyExc_ValueError, e.what());
    return -1;
  }

  return 0;
}

// Deleting a streaming algorithm is not as simple as deleting the pointer. If
// the algorithm is configured to be a generator algorithm (has only its
// sources connected), then we need to call deleteNetwork on that algorithm.
// If the algorithm is not connected to anything, we can safely delete it (we
// can also call deleteNetwork on it). If
// the algorithm has either both its sources and sinks connected, or just its
// sinks, do not delete it, because it is connected to a network and will be
// eventually deleted with the deleteNetwork function.
void PyStreamingAlgorithm::tp_dealloc (PyObject* obj) {
  PyStreamingAlgorithm* self = reinterpret_cast<PyStreamingAlgorithm*>(obj);
  // FIXME: need to deallocate something here I guess...
  //if (self->isGenerator) streaming::deleteNetwork(self->algo);

  self->ob_type->tp_free(obj);
}

PyObject* PyStreamingAlgorithm::configure (PyStreamingAlgorithm* self, PyObject* args, PyObject* keywds) {
  E_DEBUG(EPyBindings, PY_ALGONAME << "::Configure()");

  // create the list of named parameters that this algorithm can accept
  ParameterMap pm = self->algo->defaultParameters();

  // parse parameters
  try {
    parseParameters(&pm, args, keywds);
  }
  catch (const std::exception& e) {
    ostringstream msg;
    msg << "Error while parsing parameters: " << e.what();
    PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
    return NULL;
  }

  // actually configure the underlying C++ algorithm
  try {
    self->algo->configure(pm);
  }
  catch (std::exception& e) {
    ostringstream msg;
    msg << "Error while configuring " << self->algo->name() << ": " << e.what();
    PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
    return NULL;
  }

  E_DEBUG(EPyBindings, PY_ALGONAME << "::Configure() done!");

  Py_RETURN_NONE;
}

PyObject* PyStreamingAlgorithm::hasSink(PyStreamingAlgorithm* self, PyObject* obj) {
  char* name = PyString_AsString(obj);
  if (name == NULL) {
    PyErr_SetString(PyExc_ValueError, "Algorithm.hasSink requires 1 string argument");
    return NULL;
  }

  bool result = contains(self->algo->inputs(), name);
  return toPython((void*)&result, BOOL);
}

PyObject* PyStreamingAlgorithm::hasSource(PyStreamingAlgorithm* self, PyObject* obj) {
  char* name = PyString_AsString(obj);
  if (name == NULL) {
    PyErr_SetString(PyExc_ValueError, "Algorithm.hasSource requires 1 string argument");
    return NULL;
  }

  bool result = contains(self->algo->outputs(), name);
  return toPython((void*)&result, BOOL);
}


PyObject* PyStreamingAlgorithm::push(PyStreamingAlgorithm* self, PyObject* args) {
  vector<PyObject*> argsV = unpack(args);
  if (argsV.size() != 2) {
    PyErr_SetString(PyExc_ValueError, "Algorithm.push requires 2 arguments");
    return NULL;
  }

  if (!PyString_Check(argsV[0])) {
    PyErr_SetString(PyExc_ValueError, "Algorithm.push requires a string as the first argument");
    return NULL;
  }

  // name of the source to push onto
  string name = PyString_AS_STRING(argsV[0]);

  // check if source exists
  if (!contains(self->algo->outputs(), name)) {
    ostringstream msg;
    msg << self->algo->name() << " does not have an input named " << name;
    PyErr_SetString(PyExc_ValueError, msg.str().c_str());
    return NULL;
  }

  SourceBase& src = self->algo->output(name);

  // edt of given data
  Edt tp = typeInfoToEdt(src.typeInfo());

  #define PUSH_COPY(tname, type) { \
    type* val = (type*)tname::fromPythonCopy(argsV[1]); \
    src.push(*val); \
    delete val; \
    break; \
  }

  switch(tp) {
    case INTEGER:             PUSH_COPY(Integer, int);
    case REAL:                PUSH_COPY(PyReal, Real);
    case STRING:              PUSH_COPY(String, string);
    case STEREOSAMPLE:        PUSH_COPY(PyStereoSample, StereoSample);
    case VECTOR_STRING:       PUSH_COPY(VectorString, vector<string>);
    case VECTOR_STEREOSAMPLE: PUSH_COPY(VectorStereoSample, vector<StereoSample>);

    case VECTOR_REAL: {
      RogueVector<Real>* val = (RogueVector<Real>*)VectorReal::fromPythonRef(argsV[1]);
      src.push(*(vector<Real>*)val);
      delete val;
      break;
    }

    default:
      ostringstream msg;
      msg << "given value type not supported: " << edtToString(tp);
      PyErr_SetString(PyExc_ValueError, msg.str().c_str()); return NULL;
  }

  #undef PUSH_COPY

  Py_RETURN_NONE;
}

PyObject* PyStreamingAlgorithm::getInputType(PyStreamingAlgorithm* self, PyObject* obj) {
  char* name = PyString_AsString(obj);
  if (name == NULL) {
    PyErr_SetString(PyExc_TypeError, "Algorithm.getInputType requires 1 string argument");
    return NULL;
  }

  string inputName = name;

  if (!contains(self->algo->inputs(), inputName)) {
    ostringstream msg;
    msg << "'" << inputName << "' is not an input of " << self->algo->name() << ". Available inputs are " << self->algo->inputNames();
    PyErr_SetString(PyExc_ValueError, msg.str().c_str());
    return NULL;
  }

  string result = edtToString( typeInfoToEdt( self->algo->input(inputName).typeInfo() ) );
  return toPython((void*)&result, STRING);
}

PyObject* PyStreamingAlgorithm::getOutputType(PyStreamingAlgorithm* self, PyObject* obj) {
  char* name = PyString_AsString(obj);
  if (name == NULL) {
    PyErr_SetString(PyExc_TypeError, "Algorithm.getOutputType requires 1 string argument");
    return NULL;
  }

  string outputName = name;

  if (!contains(self->algo->outputs(), outputName)) {
    ostringstream msg;
    msg << "'" << outputName << "' is not an output of " << self->algo->name() << ". Available outputs are " << self->algo->outputNames();
    PyErr_SetString(PyExc_ValueError, msg.str().c_str());
    return NULL;
  }

  string result = edtToString( typeInfoToEdt( self->algo->output(outputName).typeInfo() ) );
  return toPython((void*)&result, STRING);
}

PyObject* PyStreamingAlgorithm::paramType(PyStreamingAlgorithm* self, PyObject* obj) {
  if (!PyString_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "expected string as argument");
    return NULL;
  }

  string name = PyString_AsString(obj);

  // check if parameter exists
  if (self->algo->defaultParameters().find(name) == self->algo->defaultParameters().end()) {
    ostringstream msg;
    msg << "'" << name << "' is not a parameter of " << self->algo->name();
    PyErr_SetString(PyExc_ValueError, msg.str().c_str());
    return NULL;
  }

  string tp = edtToString(paramTypeToEdt(self->algo->parameter(name).type()));

  return toPython((void*)&tp, STRING);
}


PyObject* PyStreamingAlgorithm::paramValue(PyStreamingAlgorithm* self, PyObject* obj) {
  if (!PyString_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "expected string as argument");
    return NULL;
  }

  string name = PyString_AS_STRING(obj);

  // check if parameter exists
  if (self->algo->defaultParameters().find(name) == self->algo->defaultParameters().end()) {
    ostringstream msg;
    msg << "'" << name << "' is not a parameter of " << self->algo->name();
    PyErr_SetString(PyExc_ValueError, msg.str().c_str());
    return NULL;
  }

  // determine the type of the parameter and return the appropriate Python object
  PyObject* result = paramToPython( self->algo->parameter(name) );

  if (result == NULL) {
    // param not configured
    Py_RETURN_NONE;
  }
  else {
    return result;
  }
}


PyObject* PyStreamingAlgorithm::getDoc(PyStreamingAlgorithm* self) {
  const AlgorithmInfo<streaming::Algorithm>& inf = streaming::AlgorithmFactory::getInfo(self->algo->name());
  string docstr = generateDocString<streaming::Algorithm>(*(self->algo), inf.description);
  return PyString_FromString(docstr.c_str());
}

PyObject* PyStreamingAlgorithm::getStruct(PyStreamingAlgorithm* self) {
  const AlgorithmInfo<streaming::Algorithm>& inf = streaming::AlgorithmFactory::getInfo(self->algo->name());
  return generateDocStruct<streaming::Algorithm>(*(self->algo), inf.description);
}

static PyMethodDef PyStreamingAlgorithm_methods[] = {
  { "name", (PyCFunction)PyStreamingAlgorithm::name, METH_NOARGS,
      "Returns the name of the algorithm." },

  { "outputNames", (PyCFunction)PyStreamingAlgorithm::outputNames, METH_NOARGS,
      "Returns a list of the source names of the algorithm." },

  { "inputNames", (PyCFunction)PyStreamingAlgorithm::inputNames, METH_NOARGS,
      "Returns a list of the sink names of the algorithm." },

  { "parameterNames", (PyCFunction)PyStreamingAlgorithm::parameterNames, METH_NOARGS,
      "Returns the names of the parameters for this algorithm." },

  { "__configure__", (PyCFunction)PyStreamingAlgorithm::configure, METH_VARARGS | METH_KEYWORDS,
      "Configures the algorithm." },

  { "hasInput", (PyCFunction)PyStreamingAlgorithm::hasSink, METH_O,
      "Returns true if algorithm contains given sink name."},

  { "hasOutput", (PyCFunction)PyStreamingAlgorithm::hasSource, METH_O,
      "Returns true if algorithm contains given source name."},

  { "push", (PyCFunction)PyStreamingAlgorithm::push, METH_VARARGS,
      "acquires 1 token for the given source name."},

  { "getInputType", (PyCFunction)PyStreamingAlgorithm::getInputType, METH_O,
      "returns a string representation of input type specified by a given name"},

  { "getOutputType", (PyCFunction)PyStreamingAlgorithm::getOutputType, METH_O,
      "returns a string representation of input type specified by a given name"},

  { "paramType", (PyCFunction)PyStreamingAlgorithm::paramType, METH_O,
      "Returns the type of the parameter given by its name" },

  { "paramValue", (PyCFunction)PyStreamingAlgorithm::paramValue, METH_O,
      "Returns the value of the parameter or None if not yet configured" },

  { "getDoc", (PyCFunction)PyStreamingAlgorithm::getDoc, METH_NOARGS,
      "Returns the doc string for the algorithm"},

  { "getStruct", (PyCFunction)PyStreamingAlgorithm::getStruct, METH_NOARGS,
      "Returns the doc struct for the algorithm"},

  { NULL } /* Sentinel */
};

static PyTypeObject PyStreamingAlgorithmType = {
  PyObject_HEAD_INIT(NULL)
  0,                                                      // ob_size
  "essentia.streaming.Algorithm",                          // tp_name
  sizeof(PyStreamingAlgorithm),                           // tp_basicsize
  0,                                                      // tp_itemsize
  PyStreamingAlgorithm::tp_dealloc,                       // tp_dealloc
  0,                                                      // tp_print
  0,                                                      // tp_getattr
  0,                                                      // tp_setattr
  0,                                                      // tp_compare
  0,                                                      // tp_repr
  0,                                                      // tp_as_number
  0,                                                      // tp_as_sequence
  0,                                                      // tp_as_mapping
  0,                                                      // tp_hash
  0,                                                      // tp_call
  0,                                                      // tp_str
  0,                                                      // tp_getattro
  0,                                                      // tp_setattro
  0,                                                      // tp_as_buffer
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,               // tp_flags
  "essentia::streaming::Algorithm wrapper objects", // tp_doc
  0,                                                      // tp_traverse
  0,                                                      // tp_clear
  0,                                                      // tp_richcompare
  0,                                                      // tp_weaklistoffset
  0,                                                      // tp_iter
  0,                                                      // tp_iternext
  PyStreamingAlgorithm_methods,                           // tp_methods
  0,                                                      // tp_members
  0,                                                      // tp_getset
  0,                                                      // tp_base
  0,                                                      // tp_dict
  0,                                                      // tp_descr_get
  0,                                                      // tp_descr_set
  0,                                                      // tp_dictoffset
  (initproc)PyStreamingAlgorithm::tp_init,                // tp_init
  0,                                                      // tp_alloc
  PyStreamingAlgorithm::tp_new,                           // tp_new
};


#undef PY_ALGONAME
