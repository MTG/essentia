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
#include "structmember.h"
#include "algorithm.h"
#include "algorithmfactory.h"
#include "roguevector.h"
#include "commonfunctions.h"
#include "parsing.h"
using namespace std;
using namespace essentia;
using namespace standard;


#define PY_ALGONAME "Standard : " << self->algo->name()


/**
 * The algorithm structure. Contains a pointer to the C++ algorithm, and a bool
 * indicating whether it has already been configured or not.
 */
class PyAlgorithm {

 public:
  PyObject_HEAD

  Algorithm* algo;

  static PyObject* make_new(PyTypeObject* type, PyObject* args, PyObject* kwds);
  static int init(PyAlgorithm *self, PyObject *args, PyObject *kwds);
  static void dealloc(PyObject* self);

  static PyObject* name(PyAlgorithm* self) {
    return String::toPythonCopy(&self->algo->name());
  }

  static PyObject* reset(PyAlgorithm* self) {
    self->algo->reset();
    Py_RETURN_NONE;
  }

  static PyObject* inputNames(PyAlgorithm* self) {
    vector<string> names = self->algo->inputNames();
    return VectorString::toPythonCopy(&names);
  }

  static PyObject* outputNames(PyAlgorithm* self) {
    vector<string> names = self->algo->outputNames();
    return VectorString::toPythonCopy(&names);
  }

  static PyObject* parameterNames(PyAlgorithm* self) {
    vector<string> names = self->algo->defaultParameters().keys();
    return VectorString::toPythonCopy(&names);
  }

  static PyObject* configure(PyAlgorithm* self, PyObject* args, PyObject* keywds);
  static PyObject* compute(PyAlgorithm* self, PyObject* args);
  static PyObject* inputType(PyAlgorithm* self, PyObject* name);
  static PyObject* paramType(PyAlgorithm* self, PyObject* name);
  static PyObject* paramValue(PyAlgorithm* self, PyObject* name);

  static PyObject* getDoc(PyAlgorithm* self);
  static PyObject* getStruct(PyAlgorithm* self);
};


PyObject* PyAlgorithm::make_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
  return (PyObject*)(type->tp_alloc(type, 0));
}

void PyAlgorithm::dealloc(PyObject* self) {
  delete ((PyAlgorithm*)self)->algo;
  self->ob_type->tp_free(self);
}


int PyAlgorithm::init(PyAlgorithm *self, PyObject *args, PyObject *kwds) {
  static char *kwlist[] = { (char*)"name", NULL };
  char* algoname;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &algoname)) {
    return -1;
  }

  E_DEBUG(EPyBindings, "Standard : " << algoname << "::init()");

  try {
    E_DEBUG(EPyBindings, "Standard : creating with name " << algoname);
    self->algo = AlgorithmFactory::create(algoname);
  }
  catch (exception& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
    return -1;
  }

  E_DEBUG(EPyBindings, PY_ALGONAME << "::init() done!");
  return 0;
}



PyObject* PyAlgorithm::configure(PyAlgorithm* self, PyObject* args, PyObject* keywds) {

  E_DEBUG(EPyBindings, PY_ALGONAME << "::configure()");

  // create the list of named parameters that this algorithm can accept
  ParameterMap pm = self->algo->defaultParameters();

  // parse parameters
  try {
    parseParameters(&pm, args, keywds);
  }
  catch (const exception& e) {
    ostringstream msg;
    msg << "Error while parsing parameters: " << e.what();
    PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
    return NULL;
  }

  // actually configure the underlying C++ algorithm
  try {
    self->algo->configure(pm);
  }
  catch (const exception& e) {
    ostringstream msg;
    msg << "Error while configuring " << self->algo->name() << ": " << e.what();
    PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
    return NULL;
  }

  E_DEBUG(EPyBindings, PY_ALGONAME << "::configure() done!");

  Py_RETURN_NONE;
}


void deallocate_inputs(vector<void*> inputs, vector<Edt> inputTypes) {
  if (inputs.size() != inputTypes.size()) {
    throw EssentiaException("PyAlgorithm: deallocate_outputs requires vector arguments of equal length");
  }

  for (int i=0; i<int(inputs.size()); ++i) {
    Edt tp = inputTypes[i];
    if (tp != POOL) dealloc(inputs[i], tp);
  }
}


void deallocate_outputs(vector<void*> outputs, vector<Edt> outputTypes) {
  if (outputs.size() != outputTypes.size()) {
    throw EssentiaException("PyAlgorithm: deallocate_outputs requires vector arguments of equal length");
  }

  for (int i=0; i<int(outputs.size()); ++i) {
    if (outputs[i] == NULL) continue;
    Edt tp = outputTypes[i];
    if (tp != VECTOR_REAL && tp != VECTOR_COMPLEX && tp != VECTOR_INTEGER &&
        tp != MATRIX_REAL && tp != POOL) {
      dealloc(outputs[i], tp);
    }
  }
}


PyObject* PyAlgorithm::compute(PyAlgorithm* self, PyObject* args) {
  E_DEBUG(EPyBindings, PY_ALGONAME << "::compute()");

  // parse the arguments into separate python objects
  vector<PyObject*> arg_list = unpack(args);

  int nInputs = self->algo->inputs().size();
  vector<string> inputNames = self->algo->inputNames();
  vector<const type_info*> inputTypes = self->algo->inputTypes();

  // check the correct number of arguments has been given to the function call
  if (int(arg_list.size()) != nInputs) {
    ostringstream msg;
    msg << self->algo->name() << ".compute has " << nInputs << " inputs, " << arg_list.size() << " given";
    PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
    return NULL;
  }

  // bind the inputs and outputs

  // parse all inputs given by the python interpreter to the corresponding
  // C++ variables and assign them to the algorithm's input ports
  E_DEBUG_NONL(EPyBindings, PY_ALGONAME << ": binding inputs...");

  // parse each python obj to a Cpp pointer and set appropriate input
  vector<void*> givenInputs(nInputs);
  vector<Edt> givenInputTypes(nInputs);

  for (int i=0; i<nInputs; ++i) {
    InputBase& port = self->algo->input(inputNames[i]);
    Edt tp = typeInfoToEdt(*inputTypes[i]);
    givenInputTypes[i] = tp;

    #define SET_PORT_COPY(tname, type) \
      givenInputs[i] = tname::fromPythonCopy(arg_list[i]); \
      port.set(*(type*)givenInputs[i]); \
      break;

    #define SET_PORT_REF(tname, type) \
      givenInputs[i] = tname::fromPythonRef(arg_list[i]); \
      port.set(*(type*)givenInputs[i]); \
      break;

    try {
      switch (tp) {
        case REAL:                 SET_PORT_COPY(PyReal, Real);
        case VECTOR_STRING:        SET_PORT_COPY(VectorString, vector<string>);
        case STRING:               SET_PORT_COPY(String, string);
        case BOOL:                 SET_PORT_COPY(Boolean, bool);
        case INTEGER:              SET_PORT_COPY(Integer, int);
        case VECTOR_VECTOR_REAL:   SET_PORT_COPY(VectorVectorReal, vector<vector<Real> >);
        case VECTOR_VECTOR_STRING: SET_PORT_COPY(VectorVectorString, vector<vector<string> >);
        case VECTOR_STEREOSAMPLE:  SET_PORT_COPY(VectorStereoSample, vector<StereoSample>);
        case MATRIX_REAL:          SET_PORT_COPY(MatrixReal, TNT::Array2D<Real>);

        case POOL:                 SET_PORT_REF(PyPool, Pool);
        case VECTOR_REAL:          SET_PORT_REF(VectorReal, vector<Real>);
        case VECTOR_INTEGER:       SET_PORT_REF(VectorInteger, vector<int>);
        case VECTOR_COMPLEX:       SET_PORT_REF(VectorComplex, vector<complex<Real> >);

        default:
          ostringstream msg;
          msg << "Error: unsupported input type: " << edtToString(tp);
          PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
          return NULL;
      }
    }
    catch (const exception& e) {
      ostringstream msg;
      msg << "an error while parsing input arguments: " << e.what() << endl;
      msg << "Please make sure the arguments are in the correct order and of the correct type";
      PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
      return NULL;
    }

    #undef SET_PORT_COPY
    #undef SET_PORT_REF
  }
  E_DEBUG(EPyBindings, "done!");

  // allocate memory for all C++ output variables for the algorithm and assign
  // them accordingly
  E_DEBUG_NONL(EPyBindings, PY_ALGONAME << ": binding outputs...");

  int nOutputs = self->algo->outputs().size();
  vector<const type_info*> outputTypeInfos = self->algo->outputTypes();
  vector<string> outputNames = self->algo->outputNames();
  vector<void*> outputs(nOutputs, (void*)NULL);
  vector<Edt> outputTypes(nOutputs);

  for (int i=0; i<nOutputs; i++) {
    OutputBase& port = self->algo->output(outputNames[i]);
    outputTypes[i] = typeInfoToEdt(*outputTypeInfos[i]);

    #define SET_PORT(type)                                                   \
      outputs[i] = (void*)new type;                                          \
      port.set(*(type*)outputs[i]);                                          \
      break;

    switch (outputTypes[i]) {
      case REAL: SET_PORT(Real);
      case STRING: SET_PORT(string);
      case BOOL: SET_PORT(bool);
      case INTEGER: SET_PORT(int);
      case STEREOSAMPLE: SET_PORT(StereoSample);
      case VECTOR_REAL:
        outputs[i] = (void*)new RogueVector<Real>((uint)10, 0.);
        reinterpret_cast<vector<Real>*>(outputs[i])->clear();
        port.set(*(vector<Real>*)outputs[i]);
        break;
        //SET_PORT(vector<Real>);
      case VECTOR_INTEGER:
        outputs[i] = (void*)new RogueVector<int>((uint)10, 3);
        reinterpret_cast<vector<Real>*>(outputs[i])->clear();
        port.set(*(vector<Real>*)outputs[i]);
        break;
        //SET_PORT(vector<int>);
      case VECTOR_COMPLEX:
        outputs[i] = (void*)new RogueVector<complex<Real> >((uint)10, 3);
        reinterpret_cast<vector<complex<Real> >*>(outputs[i])->clear();
        port.set(*(vector<complex<Real> >*)outputs[i]);
        break;
        //SET_PORT(vector<complex<Real> >);
      case VECTOR_STRING: SET_PORT(vector<string>);
      case VECTOR_STEREOSAMPLE: SET_PORT(vector<StereoSample>);
      case VECTOR_VECTOR_REAL: SET_PORT(vector<vector<Real> >);
      case VECTOR_VECTOR_STRING: SET_PORT(vector<vector<string> >);
      case MATRIX_REAL: SET_PORT(TNT::Array2D<Real>);
      case POOL: SET_PORT(Pool);

      default:
        ostringstream msg;
        msg << "In " << self->algo->name();
        msg << ".compute: unable to convert cpp type to python type: ";
        msg << edtToString(outputTypes[i]);
        PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());
        deallocate_inputs(givenInputs, givenInputTypes);
        for (int i=0; i<int(outputs.size()); ++i) {
          if (outputs[i] == NULL) continue;
          ::dealloc(outputs[i], outputTypes[i]);
        }
        return NULL;
    }

    #undef SET_PORT
  }
  E_DEBUG(EPyBindings, "done!");


  // now that the algorithm and ready and set to go (all inputs and outputs
  // are correctly bound), we can safely call the compute() method.
  E_DEBUG(EPyBindings, PY_ALGONAME << ": computing...");

  try {
    self->algo->compute();
  }
  catch (const exception& e) {
    ostringstream msg;
    msg << "In " << self->algo->name() << ".compute: " << e.what();
    PyErr_SetString(PyExc_RuntimeError, msg.str().c_str());

    // clean up temp vars
    deallocate_inputs(givenInputs, givenInputTypes);

    for (int i=0; i<int(outputs.size()); ++i) {
      if (outputs[i] == NULL) continue;
      ::dealloc(outputs[i], outputTypes[i]);
    }

    return NULL;
  }
  E_DEBUG(EPyBindings, PY_ALGONAME << ": done!");


  // now that the processing is done, convert the results back to python
  // variables and return them to the interpreter
  vector<PyObject*> result(nOutputs);

  for (int i=0; i<nOutputs; i++) {
    try {
      result[i] = toPython(outputs[i], outputTypes[i]);
    }
    catch (const exception& e) {
      // clean up
      deallocate_inputs(givenInputs, givenInputTypes);
      deallocate_outputs(outputs, outputTypes);

      ostringstream msg;
      msg << "In " << self->algo->name();
      msg << ".compute: error while converting outputs to python variables: ";
      msg << e.what();
      PyErr_SetString(PyExc_TypeError, msg.str().c_str());
      return NULL;
    }
  }

  deallocate_inputs(givenInputs, givenInputTypes);
  deallocate_outputs(outputs, outputTypes);

  E_DEBUG(EPyBindings, PY_ALGONAME << ": returning tuple with " << nOutputs << " outputs");
  E_DEBUG(EPyBindings, PY_ALGONAME << "::compute() done!");

  return buildReturnValue(result);
}


PyObject* PyAlgorithm::inputType(PyAlgorithm* self, PyObject* obj) {
  if (!PyString_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "Algorithm.inputType expects a string as the only argument");
    return NULL;
  }

  string name = PyString_AsString(obj);

  try {
    self->algo->input(name);
  }
  catch (const EssentiaException&) {
    ostringstream msg;
    msg << "'" << name << "' is not an input of " << self->algo->name() << ". Available inputs are " << self->algo->inputNames();
    PyErr_SetString(PyExc_ValueError, msg.str().c_str());
    return NULL;
  }

  string tp = edtToString( typeInfoToEdt( self->algo->input(name).typeInfo() ) );

  return String::toPythonCopy(&tp);
}


PyObject* PyAlgorithm::paramType(PyAlgorithm* self, PyObject* obj) {
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

  return String::toPythonCopy(&tp);
}

PyObject* PyAlgorithm::paramValue(PyAlgorithm* self, PyObject* obj) {
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


PyObject* PyAlgorithm::getDoc(PyAlgorithm* self) {
  const AlgorithmInfo<Algorithm>& inf = AlgorithmFactory::getInfo(self->algo->name());
  string docstr = generateDocString<Algorithm>(*(self->algo), inf.description);
  return PyString_FromString(docstr.c_str());
}


PyObject* PyAlgorithm::getStruct(PyAlgorithm* self) {
  const AlgorithmInfo<Algorithm>& inf = AlgorithmFactory::getInfo(self->algo->name());
  return generateDocStruct<Algorithm>(*(self->algo), inf.description);
}

static PyMethodDef PyAlgorithm_methods[] = {
  { "name",           (PyCFunction)PyAlgorithm::name, METH_NOARGS,
                      "Returns the name of the algorithm." },
  { "inputNames",     (PyCFunction)PyAlgorithm::inputNames, METH_NOARGS,
                      "Returns the names of the inputs of the algorithm." },
  { "outputNames",    (PyCFunction)PyAlgorithm::outputNames, METH_NOARGS,
                      "Returns the names of the outputs of the algorithm." },
  { "parameterNames", (PyCFunction)PyAlgorithm::parameterNames, METH_NOARGS,
                      "Returns the names of the parameters for this algorithm." },
  { "reset",          (PyCFunction)PyAlgorithm::reset, METH_NOARGS,
                      "Reset the algorithm to its initial state (if any)." },
  { "__configure__",  (PyCFunction)PyAlgorithm::configure, METH_VARARGS | METH_KEYWORDS,
                      "Configure the algorithm" },
  { "__compute__",    (PyCFunction)PyAlgorithm::compute, METH_VARARGS,
                      "compute the algorithm" },
  { "inputType",      (PyCFunction)PyAlgorithm::inputType, METH_O,
                      "Returns the type of the input given by its name" },
  { "paramType",      (PyCFunction)PyAlgorithm::paramType, METH_O,
                      "Returns the type of the parameter given by its name" },
  { "paramValue",     (PyCFunction)PyAlgorithm::paramValue, METH_O,
                      "Returns the value of the parameter or None if not yet configured" },
  { "getDoc",         (PyCFunction)PyAlgorithm::getDoc, METH_NOARGS,
                      "Returns the doc string for the algorithm"},
  { "getStruct",      (PyCFunction)PyAlgorithm::getStruct, METH_NOARGS,
                      "Returns the doc struct for the algorithm"},
  { NULL }  /* Sentinel */
};

static PyTypeObject PyAlgorithmType = {
    PyObject_HEAD_INIT(NULL)
    0,                                                    // ob_size
    "essentia.standard.Algorithm",                                 // tp_name
    sizeof(PyAlgorithm),                                  // tp_basicsize
    0,                                                    // tp_itemsize
    PyAlgorithm::dealloc,                                 // tp_dealloc
    0,                                                    // tp_print
    0,                                                    // tp_getattr
    0,                                                    // tp_setattr
    0,                                                    // tp_compare
    0,                                                    // tp_repr
    0,                                                    // tp_as_number
    0,                                                    // tp_as_sequence
    0,                                                    // tp_as_mapping
    0,                                                    // tp_hash
    0,                                                    // tp_call
    0,                                                    // tp_str
    0,                                                    // tp_getattro
    0,                                                    // tp_setattro
    0,                                                    // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,             // tp_flags
    "essentia::standard::Algorithm wrapper objects",                // tp_doc
    0,                                                    // tp_traverse
    0,                                                    // tp_clear
    0,                                                    // tp_richcompare
    0,                                                    // tp_weaklistoffset
    0,                                                    // tp_iter
    0,                                                    // tp_iternext
    PyAlgorithm_methods,                                  // tp_methods
    0,                                                    // tp_members
    0,                                                    // tp_getset
    0,                                                    // tp_base
    0,                                                    // tp_dict
    0,                                                    // tp_descr_get
    0,                                                    // tp_descr_set
    0,                                                    // tp_dictoffset
    (initproc)PyAlgorithm::init,                          // tp_init
    0,                                                    // tp_alloc
    PyAlgorithm::make_new,                                // tp_new
};


#undef PY_ALGONAME
