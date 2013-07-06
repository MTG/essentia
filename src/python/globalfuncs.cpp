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

#include "essentiamath.h" // for silenceCutoff
#include "streamingalgorithm.h"
#include "poolstorage.h" // connecting pools
#include "../algorithms/io/fileoutputproxy.h" // connecting FileOutput algorithm
#include "bpmutil.h" // postProcessTicks()

static PyObject*
get_version() {
  return PyString_FromString(version);
}

static PyObject*
debug_level() {
  return PyInt_FromLong(activatedDebugLevels);
}

static PyObject*
set_debug_level(PyObject* self, PyObject* arg) {
  if (!PyInt_Check(arg) && !PyLong_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)"argument must be an integer");
    return NULL;
  }

  long dbgLevels = PyInt_AsLong(arg);
  activatedDebugLevels = dbgLevels;

  Py_RETURN_NONE;
}

static PyObject* log_debug(PyObject* notUsed, PyObject* args) {
  // parse args to get Source alg, name and source alg and source name
  vector<PyObject*> argsV = unpack(args);
  if (argsV.size() != 2 ||
      (!PyInt_Check(argsV[0]) && !PyLong_Check(argsV[0])) ||
      !PyString_Check(argsV[1])) {
    PyErr_SetString(PyExc_ValueError, "expecting arguments (DebugLevel, string)");
    return NULL;
  }

  long dbgLevel = PyInt_AsLong(argsV[0]);

  E_DEBUG((DebuggingModule)dbgLevel, PyString_AS_STRING(argsV[1]));
  Py_RETURN_NONE;
}

static PyObject*
info_level() {
  if (essentia::infoLevelActive) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

static PyObject*
set_info_level(PyObject* self, PyObject* arg) {
  if (arg == Py_True) {
    essentia::infoLevelActive = true;
    Py_RETURN_NONE;
  }
  if (arg == Py_False) {
    essentia::infoLevelActive = false;
    Py_RETURN_NONE;
  }

  PyErr_SetString(PyExc_TypeError, (char*)"argument needs to be either True or False");
  return NULL;
}

static PyObject*
log_info(PyObject* self, PyObject* arg) {
  if (!PyString_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)"argument must be a string");
    return NULL;
  }

  E_INFO(PyString_AS_STRING(arg));
  Py_RETURN_NONE;
}

static PyObject*
warning_level() {
  if (essentia::warningLevelActive) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

static PyObject*
set_warning_level(PyObject* self, PyObject* arg) {
  if (arg == Py_True) {
    essentia::warningLevelActive = true;
    Py_RETURN_NONE;
  }
  if (arg == Py_False) {
    essentia::warningLevelActive = false;
    Py_RETURN_NONE;
  }

  PyErr_SetString(PyExc_TypeError, (char*)"argument needs to be either True or False");
  return NULL;
}

static PyObject*
log_warning(PyObject* self, PyObject* arg) {
  if (!PyString_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)"argument must be a string");
    return NULL;
  }

  E_WARNING(PyString_AS_STRING(arg));
  Py_RETURN_NONE;
}

static PyObject*
error_level() {
  if (essentia::errorLevelActive) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

static PyObject*
set_error_level(PyObject* self, PyObject* arg) {
  if (arg == Py_True) {
    essentia::errorLevelActive = true;
    Py_RETURN_NONE;
  }
  if (arg == Py_False) {
    essentia::errorLevelActive = false;
    Py_RETURN_NONE;
  }

  PyErr_SetString(PyExc_TypeError, (char*)"argument needs to be either True or False");
  return NULL;
}

static PyObject*
log_error(PyObject* self, PyObject* arg) {
  if (!PyString_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)"argument must be a string");
    return NULL;
  }

  E_ERROR(PyString_AS_STRING(arg));
  Py_RETURN_NONE;
}


/**
  * normalize helper function for arrays
  */
static PyObject*
normalize(PyObject* self, PyObject* arg) {
  if (!PyArray_Check(arg) || PyList_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)" requires argument types:numpy array or list");
    return NULL;
  }
  vector<Real>* array = reinterpret_cast<vector<Real>*>(VectorReal::fromPythonRef(arg));
  essentia::normalize(*array);
  RogueVector<Real>* result = new RogueVector<Real>(array->size(), 0.);
  for (int i=0; i<int(array->size()); ++i) { (*result)[i] = (*array)[i]; }
  return VectorReal::toPythonRef(result);
}

/**
  * derivative helper function for arrays
  */
static PyObject*
derivative(PyObject* self, PyObject* arg) {
  if (!PyArray_Check(arg) || PyList_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)" requires argument types:numpy array or list");
    return NULL;
  }
  vector<Real>* array = reinterpret_cast<vector<Real>*>(VectorReal::fromPythonRef(arg));
  vector<Real> derivative = essentia::derivative(*array);
  RogueVector<Real>* result = new RogueVector<Real>(derivative.size(), 0.);
  for (int i=0; i<int(array->size()); ++i) { (*result)[i] = derivative[i]; }
  return VectorReal::toPythonRef(result);
}

/**
 * Helper function returning the instant_power as a double to avoid double
 * conversion in is_silent.
 */
static double
internal_instant_power(PyObject* obj) {
  double p = 0;
  for (int i=0; i<PyArray_SIZE(obj); i++) {
    double x = ((float*)PyArray_DATA(obj))[i];
    p += x * x;
  }
  p /= PyArray_SIZE(obj);
  return p;
}



/**
 * Returns the instant_power.
 */
static PyObject*
instant_power(PyObject* self, PyObject* arg) {
  if (!PyArray_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)"argument has to be a numpy array");
    return NULL;
  }

  return PyFloat_FromDouble(internal_instant_power(arg));
}

/**
 * Returns true if the frame is silent.
 */
static PyObject*
is_silent(PyObject* self, PyObject* arg) {
  if (!PyArray_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)"argument has to be a numpy array");
    return NULL;
  }

  double p = internal_instant_power(arg);

  if (p < silenceCutoff) Py_RETURN_TRUE;
  else                   Py_RETURN_FALSE;
}

static PyObject*
is_power_two(PyObject* notUsed, PyObject* arg) {
  if (!PyInt_Check(arg) && !PyLong_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)"argument must be an integer");
    return NULL;
  }

  bool b = isPowerTwo(PyInt_AsLong(arg));

  if (b) Py_RETURN_TRUE;
  else   Py_RETURN_FALSE;
}

static PyObject*
next_power_two(PyObject* notUsed, PyObject* arg) {
  if (!PyFloat_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)"argument must be an integer");
    return NULL;
  }

  return PyInt_FromLong(nextPowerTwo(int(PyFloat_AS_DOUBLE(arg))));
}

static PyObject*
linToDb(PyObject* notUsed, PyObject* arg) {
  if (!PyFloat_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)"argument must be a float");
    return NULL;
  }

  Real db = lin2db( Real( PyFloat_AS_DOUBLE(arg) ) );
  return PyFloat_FromDouble( double(db) );
}


static PyObject*
dbToLin(PyObject* notUsed, PyObject* arg) {
  if (!PyFloat_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)"argument must be a float");
    return NULL;
  }

  Real lin = db2lin( Real( PyFloat_AS_DOUBLE(arg) ) );
  return PyFloat_FromDouble( double(lin) );
}

static PyObject*
powToDb(PyObject* notUsed, PyObject* arg) {
  if (!PyFloat_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)"argument must be a float");
    return NULL;
  }

  Real db = pow2db( Real( PyFloat_AS_DOUBLE(arg) ) );
  return PyFloat_FromDouble( double(db) );
}

static PyObject*
dbToPow(PyObject* notUsed, PyObject* arg) {
  if (!PyFloat_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)"argument must be a float");
    return NULL;
  }

  Real lin = db2pow( Real( PyFloat_AS_DOUBLE(arg) ) );
  return PyFloat_FromDouble( double(lin) );
}

static PyObject*
dbToAmp(PyObject* notUsed, PyObject* arg) {
  if (!PyFloat_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)"argument must be a float");
    return NULL;
  }

  Real lin = db2amp( Real( PyFloat_AS_DOUBLE(arg) ) );
  return PyFloat_FromDouble( double(lin) );
}

static PyObject*
ampToDb(PyObject* notUsed, PyObject* arg) {
  if (!PyFloat_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)"argument must be a float");
    return NULL;
  }

  Real db = amp2db( Real( PyFloat_AS_DOUBLE(arg) ) );
  return PyFloat_FromDouble( double(db) );
}

static PyObject*
barkToHz(PyObject* notUsed, PyObject* arg) {
  if (!PyFloat_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)"argument must be a float");
    return NULL;
  }

  Real hz = bark2hz( Real( PyFloat_AS_DOUBLE(arg) ) );
  return PyFloat_FromDouble( double(hz) );
}


static PyObject*
hzToBark(PyObject* notUsed, PyObject* arg) {
  if (!PyFloat_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)"argument must be a float");
    return NULL;
  }

  Real bark = hz2bark( Real( PyFloat_AS_DOUBLE(arg) ) );
  return PyFloat_FromDouble( double(bark) );
}

static PyObject*
melToHz(PyObject* notUsed, PyObject* arg) {
  if (!PyFloat_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)"argument must be a float");
    return NULL;
  }

  Real hz = mel2hz( Real( PyFloat_AS_DOUBLE(arg) ) );
  return PyFloat_FromDouble( double(hz) );
}


static PyObject*
hzToMel(PyObject* notUsed, PyObject* arg) {
  if (!PyFloat_Check(arg)) {
    PyErr_SetString(PyExc_TypeError, (char*)"argument must be a float");
    return NULL;
  }

  Real mel = hz2mel( Real( PyFloat_AS_DOUBLE(arg) ) );
  return PyFloat_FromDouble( double(mel) );
}

template <typename T>
PyObject* algorithmInfo(T* algo) {
  PyObject* result = PyDict_New();

  vector<string> info(3, "");
  PyObject* sublist = PyList_New(0);

  // add general description & version number
  AlgorithmInfo<T> staticInfo = EssentiaFactory<T>::getInfo(algo->name());

  PyDict_SetItemString(result, "description", PyString_FromString(staticInfo.description.c_str()));

  // add inputs
  typename T::InputMap inputs = algo->inputs();
  for (typename T::InputMap::const_iterator it = inputs.begin();
       it != inputs.end();
       ++it) {
    string name = it->first;
    info[0] = name;
    info[1] = edtToString( typeInfoToEdt( it->second->typeInfo() ) );
    info[2] = algo->inputDescription[name];

    PyList_Append(sublist, VectorString::toPythonCopy(&info));
  }
  PyDict_SetItemString(result, "inputs", sublist);

  // add outputs
  sublist = PyList_New(0);
  typename T::OutputMap outputs = algo->outputs();
  for (typename T::OutputMap::const_iterator it = outputs.begin();
       it != outputs.end();
       ++it) {
    string name = it->first;
    info[0] = name;
    info[1] = edtToString( typeInfoToEdt( it->second->typeInfo() ) );
    info[2] = algo->outputDescription[name];

    PyList_Append(sublist, VectorString::toPythonCopy(&info));
  }
  PyDict_SetItemString(result, "outputs", sublist);

  // add parameters
  sublist = PyList_New(0);
  info.resize(4);
  ParameterMap pm = algo->defaultParameters();

  for (ParameterMap::const_iterator it=pm.begin(); it != pm.end(); ++it) {
    string name = it->first;
    info[0] = name;
    info[1] = algo->parameterDescription[name];
    info[2] = algo->parameterRange[name];
    if (it->second.isConfigured()) {
      info[3] = it->second.toString();
    }
    else {
      info[3] = "";
    }

    PyList_Append(sublist, VectorString::toPythonCopy(&info));
  }
  PyDict_SetItemString(result, "parameters", sublist);

  delete algo;

  return result;
}


static PyObject* streaming_info(PyObject* self, PyObject* args) {
  char* name;
  if (!PyArg_ParseTuple(args, "s", &name))
    return NULL;

  streaming::Algorithm* algo;
  try {
    algo = streaming::AlgorithmFactory::create(name);
  }
  catch (exception& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
    return NULL;
  }

  return algorithmInfo(algo);
}

static PyObject* standard_info(PyObject* self, PyObject* args) {
  char* name;
  if (!PyArg_ParseTuple(args, "s", &name))
    return NULL;

  standard::Algorithm* algo;
  try {
    algo = standard::AlgorithmFactory::create(name);
  }
  catch (exception& e) {
    PyErr_SetString(PyExc_ValueError, e.what());
    return NULL;
  }

  return algorithmInfo(algo);
}

template<typename T>
static PyObject* algorithmKeys() {
  vector<string> algoNames = T::keys();
  return VectorString::toPythonCopy(&algoNames);
}

static PyObject* keys() {
  return algorithmKeys<standard::AlgorithmFactory>();
}

static PyObject* skeys() {
  return algorithmKeys<streaming::AlgorithmFactory>();
}

static PyObject* totalProduced(PyObject* notUsed, PyObject* args) {
  // parse args to get Source alg, name and source alg and source name
  vector<PyObject*> argsV = unpack(args);
  if (argsV.size() != 2 ||
      (!PyType_IsSubtype(argsV[0]->ob_type, &PyStreamingAlgorithmType) ||
      !PyString_Check(argsV[1]))) {
    PyErr_SetString(PyExc_ValueError, "expecting arguments (streaming.Algorithm alg, str sourcename)");
    return NULL;
  }
  int result = 0;
  PyStreamingAlgorithm* sourceAlg = reinterpret_cast<PyStreamingAlgorithm*>(argsV[0]);
  string sourceName = string(PyString_AS_STRING(argsV[1]));
  try {
   result = sourceAlg->algo->outputs()[sourceName]->totalProduced();
  }
  catch (const exception& e) {
    PyErr_SetString(PyExc_TypeError, e.what());
    return NULL;
  }
  return toPython((void*)&result, INTEGER);;
}

static PyObject* connect(PyObject* notUsed, PyObject* args) {
  // parse args to get Source alg and name and Sink alg and name
  vector<PyObject*> argsV = unpack(args);

  if (argsV.size() != 4 ||
      (  !PyType_IsSubtype(argsV[0]->ob_type, &PyStreamingAlgorithmType) &&
         !PyType_IsSubtype(argsV[0]->ob_type, &PyVectorInputType)  ) ||
      !PyString_Check(argsV[1]) ||
      !PyType_IsSubtype(argsV[2]->ob_type, &PyStreamingAlgorithmType) ||
      !PyString_Check(argsV[3])) {
    PyErr_SetString(PyExc_ValueError,
                    "expecting arguments (streaming.Algorithm sourceAlg, str "
                    "sourceName, streaming.Algorithm sinkAlg, str sinkName)");
    return NULL;
  }

  PyStreamingAlgorithm* sourceAlg = reinterpret_cast<PyStreamingAlgorithm*>(argsV[0]);
  string sourceName = string(PyString_AS_STRING(argsV[1]));
  PyStreamingAlgorithm* sinkAlg = reinterpret_cast<PyStreamingAlgorithm*>(argsV[2]);
  string sinkName = string(PyString_AS_STRING(argsV[3]));

  try {
    connect(sourceAlg->algo->output(sourceName),
            sinkAlg->algo->input(sinkName));

    // make sure to set the isGenerator flag on the sinkAlg to false
    sinkAlg->isGenerator = false;
  }
  catch (const exception& e) {
    PyErr_SetString(PyExc_TypeError, e.what());
    return NULL;
  }

  Py_RETURN_NONE;
}


static PyObject* poolConnect(PyObject* notUsed, PyObject* args) {
  // parse args into (source alg, source name, pool, key name)
  vector<PyObject*> argsV = unpack(args);

  if (argsV.size() != 4 ||
      (  !PyType_IsSubtype(argsV[0]->ob_type, &PyStreamingAlgorithmType) &&
         !PyType_IsSubtype(argsV[0]->ob_type, &PyVectorInputType)  ) ||
      !PyString_Check(argsV[1]) ||
      !PyType_IsSubtype(argsV[2]->ob_type, &PyPoolType) ||
      !PyString_Check(argsV[3])) {
    PyErr_SetString(PyExc_TypeError,
                    "expecting arguments (streaming.Algorithm sourceAlg, str "
                    "sourceName, Pool sinkPool, str descriptorName");
    return NULL;
  }

  PyStreamingAlgorithm* sourceAlg = reinterpret_cast<PyStreamingAlgorithm*>(argsV[0]);
  string sourceName = string(PyString_AS_STRING(argsV[1]));
  Pool* pool = reinterpret_cast<Pool*>(PyPool::fromPythonRef(argsV[2]));
  string keyName = string(PyString_AS_STRING(argsV[3]));

  try {
    streaming::connect(sourceAlg->algo->output(sourceName), *pool, keyName);
  }
  catch (const exception& e) {
    PyErr_SetString(PyExc_TypeError, e.what());
    return NULL;
  }

  Py_RETURN_NONE;
}

static PyObject* fileOutputConnect(PyObject* notUsed, PyObject* args) {
  // parse args into (source alg, source name, pool, key name)
  vector<PyObject*> argsV = unpack(args);

  if (argsV.size() != 3 ||
      (  !PyType_IsSubtype(argsV[0]->ob_type, &PyStreamingAlgorithmType) &&
         !PyType_IsSubtype(argsV[0]->ob_type, &PyVectorInputType)  ) ||
      !PyString_Check(argsV[1]) ||
      !PyType_IsSubtype(argsV[2]->ob_type, &PyStreamingAlgorithmType)) {
    PyErr_SetString(PyExc_TypeError,
                    "expecting arguments (streaming.Algorithm sourceAlg, str "
                    "sourceName, streaming.FileOutput fileOutput");
    return NULL;
  }

  PyStreamingAlgorithm* sourceAlg = reinterpret_cast<PyStreamingAlgorithm*>(argsV[0]);
  string sourceName = string(PyString_AS_STRING(argsV[1]));

  PyStreamingAlgorithm* sinkAlg = reinterpret_cast<PyStreamingAlgorithm*>(argsV[2]);
  FileOutputProxy* fileout = dynamic_cast<FileOutputProxy*>(sinkAlg->algo);
  if (!fileout) {
    PyErr_SetString(PyExc_TypeError,
                    "It doesn't look like the algo you're trying to connect to is a FileOutputProxy...");
    return NULL;
  }

  try {
    streaming::connect(sourceAlg->algo->output(sourceName), *fileout);
  }
  catch (const exception& e) {
    PyErr_SetString(PyExc_TypeError, e.what());
    return NULL;
  }

  Py_RETURN_NONE;
}


static PyObject* nowhereConnect(PyObject* notUsed, PyObject* args) {
  // parse args into (source alg, source name)
  vector<PyObject*> argsV = unpack(args);

  if (argsV.size() != 2 ||
      (  !PyType_IsSubtype(argsV[0]->ob_type, &PyStreamingAlgorithmType) &&
         !PyType_IsSubtype(argsV[0]->ob_type, &PyVectorInputType)  ) ||
      !PyString_Check(argsV[1]) ) {
    PyErr_SetString(PyExc_TypeError,
                    "expecting arguments (streaming.Algorithm sourceAlg, "
                    "str sourceName)");
    return NULL;
  }

  PyStreamingAlgorithm* sourceAlg = reinterpret_cast<PyStreamingAlgorithm*>(argsV[0]);
  string sourceName = string(PyString_AS_STRING(argsV[1]));

  try {
    streaming::connect(sourceAlg->algo->output(sourceName), streaming::NOWHERE);
  }
  catch (const exception& e) {
    PyErr_SetString(PyExc_TypeError, e.what());
    return NULL;
  }

  Py_RETURN_NONE;
}


static PyObject* run(PyObject* notUsed, PyObject* obj) {
  if (!PyType_IsSubtype(obj->ob_type, &PyStreamingAlgorithmType) &&
      !PyType_IsSubtype(obj->ob_type, &PyVectorInputType)) {
    PyErr_SetString(PyExc_TypeError, "run must be called with a streaming algorithm");
    return NULL;
  }

  PyStreamingAlgorithm* pyAlg = reinterpret_cast<PyStreamingAlgorithm*>(obj);

  try {
    scheduler::Network(pyAlg->algo, false).run();
  }
  catch (const exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return NULL;
  }

  Py_RETURN_NONE;
}


static PyObject* reset(PyObject* notUsed, PyObject* obj) {
  if (!PyType_IsSubtype(obj->ob_type, &PyStreamingAlgorithmType) &&
      !PyType_IsSubtype(obj->ob_type, &PyVectorInputType)) {
    PyErr_SetString(PyExc_TypeError, "expected a streaming algorithm");
    return NULL;
  }

  PyStreamingAlgorithm* pyAlg = reinterpret_cast<PyStreamingAlgorithm*>(obj);

  try {
    scheduler::Network(pyAlg->algo, false).reset();
  }
  catch (const exception& e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return NULL;
  }

  Py_RETURN_NONE;
}


static PyObject* disconnect(PyObject* notUsed, PyObject* args) {
  // parse args to get Source alg and name and Sink alg and name
  vector<PyObject*> argsV = unpack(args);

  if (argsV.size() != 4 ||
      (  !PyType_IsSubtype(argsV[0]->ob_type, &PyStreamingAlgorithmType) &&
         !PyType_IsSubtype(argsV[0]->ob_type, &PyVectorInputType)  ) ||
      !PyString_Check(argsV[1]) ||
      !PyType_IsSubtype(argsV[2]->ob_type, &PyStreamingAlgorithmType) ||
      !PyString_Check(argsV[3])) {
    PyErr_SetString(PyExc_ValueError,
                    "expecting arguments (streaming.Algorithm sourceAlg, str "
                    "sourceName, streaming.Algorithm sinkAlg, str sinkName)");
    return NULL;
  }

  PyStreamingAlgorithm* sourceAlg = reinterpret_cast<PyStreamingAlgorithm*>(argsV[0]);
  string sourceName = string(PyString_AS_STRING(argsV[1]));
  PyStreamingAlgorithm* sinkAlg = reinterpret_cast<PyStreamingAlgorithm*>(argsV[2]);
  string sinkName = string(PyString_AS_STRING(argsV[3]));

  try {
    disconnect(sourceAlg->algo->output(sourceName),
               sinkAlg->algo->input(sinkName));
  }
  catch (const exception& e) {
    PyErr_SetString(PyExc_TypeError, e.what());
    return NULL;
  }

  // if this disconnection disconnected the last connection sinkAlg had to a
  // network, then sinkAlg needs to become a generator (so it can be properly
  // deleted)
  bool isStillConnected = false;
  for (streaming::Algorithm::InputMap::const_iterator it = sinkAlg->algo->inputs().begin();
       it != sinkAlg->algo->inputs().end();
       ++it) {
    if (it->second->source() != NULL) {
      isStillConnected = true;
      break;
    }
  }

  if (!isStillConnected) sinkAlg->isGenerator = true;

  Py_RETURN_NONE;
}


static PyObject* poolDisconnect(PyObject* notUsed, PyObject* args) {
  // parse args into (source alg, source name, pool, key name)
  vector<PyObject*> argsV = unpack(args);

  if (argsV.size() != 4 ||
      (  !PyType_IsSubtype(argsV[0]->ob_type, &PyStreamingAlgorithmType) &&
         !PyType_IsSubtype(argsV[0]->ob_type, &PyVectorInputType)  ) ||
      !PyString_Check(argsV[1]) ||
      !PyType_IsSubtype(argsV[2]->ob_type, &PyPoolType) ||
      !PyString_Check(argsV[3])) {
    PyErr_SetString(PyExc_TypeError,
                    "expecting arguments (streaming.Algorithm sourceAlg, str "
                    "sourceName, Pool sinkPool, str descriptorName");
    return NULL;
  }

  PyStreamingAlgorithm* sourceAlg = reinterpret_cast<PyStreamingAlgorithm*>(argsV[0]);
  string sourceName = string(PyString_AS_STRING(argsV[1]));
  Pool* pool = reinterpret_cast<Pool*>(PyPool::fromPythonRef(argsV[2]));
  string keyName = string(PyString_AS_STRING(argsV[3]));

  try {
    streaming::disconnect(sourceAlg->algo->output(sourceName), *pool, keyName);
  }
  catch (const exception& e) {
    PyErr_SetString(PyExc_TypeError, e.what());
    return NULL;
  }

  Py_RETURN_NONE;
}


static PyObject* fileOutputDisconnect(PyObject* notUsed, PyObject* args) {
  // parse args into (source alg, source name, pool, key name)
  vector<PyObject*> argsV = unpack(args);

  if (argsV.size() != 3 ||
      (  !PyType_IsSubtype(argsV[0]->ob_type, &PyStreamingAlgorithmType) &&
         !PyType_IsSubtype(argsV[0]->ob_type, &PyVectorInputType)  ) ||
      !PyString_Check(argsV[1]) ||
      !PyType_IsSubtype(argsV[2]->ob_type, &PyStreamingAlgorithmType)) {
    PyErr_SetString(PyExc_TypeError,
                    "expecting arguments (streaming.Algorithm sourceAlg, str "
                    "sourceName, streaming.FileOutput fileOutput");
    return NULL;
  }

  PyStreamingAlgorithm* sourceAlg = reinterpret_cast<PyStreamingAlgorithm*>(argsV[0]);
  string sourceName = string(PyString_AS_STRING(argsV[1]));

  PyStreamingAlgorithm* sinkAlg = reinterpret_cast<PyStreamingAlgorithm*>(argsV[2]);
  FileOutputProxy* fileout = dynamic_cast<FileOutputProxy*>(sinkAlg->algo);
  if (!fileout) {
    PyErr_SetString(PyExc_TypeError,
                    "It doesn't look like the algo you're trying to connect to is a FileOutputProxy...");
    return NULL;
  }

  try {
    sourceAlg = 0; sourceName = "", fileout = 0; // dummy line to remove unused warning
    // streaming::disconnect(*(sourceAlg->algo->outputs()[sourceName]), *fileout);
    // FIXME: implement disconnect for FileOutput
  }
  catch (const exception& e) {
    PyErr_SetString(PyExc_TypeError, e.what());
    return NULL;
  }

  Py_RETURN_NONE;
}


static PyObject* nowhereDisconnect(PyObject* notUsed, PyObject* args) {
  // parse args into (source alg, source name)
  vector<PyObject*> argsV = unpack(args);

  if (argsV.size() != 2 ||
      (  !PyType_IsSubtype(argsV[0]->ob_type, &PyStreamingAlgorithmType) &&
         !PyType_IsSubtype(argsV[0]->ob_type, &PyVectorInputType)  ) ||
      !PyString_Check(argsV[1]) ) {
    PyErr_SetString(PyExc_TypeError, "expecting arguments (streaming.Algorithm sourceAlg, str sourceName)");
    return NULL;
  }

  PyStreamingAlgorithm* sourceAlg = reinterpret_cast<PyStreamingAlgorithm*>(argsV[0]);
  string sourceName = string(PyString_AS_STRING(argsV[1]));

  try {
    streaming::disconnect(sourceAlg->algo->output(sourceName), streaming::NOWHERE);
  }
  catch (const exception& e) {
    PyErr_SetString(PyExc_TypeError, e.what());
    return NULL;
  }

  Py_RETURN_NONE;
}


// This function quickly compares two numpy matrices
static PyObject* almostEqualArray(PyObject* notUsed, PyObject* args) {
  vector<PyObject*> argv = unpack(args);

  if (argv.size() != 3 ||
      !PyArray_Check(argv[0]) || !PyArray_Check(argv[1]) ||
      PyArray_TYPE(argv[0]) != PyArray_TYPE(argv[1]) ||
      PyArray_TYPE(argv[0]) != NPY_FLOAT ||
      !PyFloat_Check(argv[2])) {
    PyErr_SetString(PyExc_TypeError, "expecting arguments (numpy.array(floats) m1, numpy.array(floats) m2, float precision)");
    return NULL;
  }

  if (PyArray_NDIM(argv[0]) != PyArray_NDIM(argv[1])) Py_RETURN_FALSE;

  if (PyArray_NDIM(argv[0]) > 2) {
    PyErr_SetString(PyExc_TypeError, "comparing numpy arrays of more than 2 dimensions not implemented");
    return NULL;
  }

  float precision = PyFloat_AS_DOUBLE(argv[2]);

  // 1-dimensional arrays
  if (PyArray_NDIM(argv[0]) == 1) {
    if (PyArray_DIM(argv[0], 0) != PyArray_DIM(argv[1], 0)) Py_RETURN_FALSE;

    for (int i=0; i<int(PyArray_DIM(argv[0], 0)); ++i) {
      Real* x = (Real*)(PyArray_BYTES(argv[0]) + i*PyArray_STRIDE(argv[0], 0));
      Real* y = (Real*)(PyArray_BYTES(argv[1]) + i*PyArray_STRIDE(argv[1], 0));
      Real diff = 0;
      if (*y == 0) diff = abs(*x);
      else if (*x == 0) diff = abs(*y);
      else {
        diff = abs(*y - *x)/abs(*y);
      }
      if (diff > precision) {
        cout << "almostEqualArray: x="<< *x << ", y=" << *y << ", difference=" << diff << " allowed precision=" << precision << endl;
        Py_RETURN_FALSE;
      }
    }
  }

  // 2-dimensional arrays
  else if (PyArray_NDIM(argv[0]) == 2) {
    if (PyArray_DIM(argv[0], 0) != PyArray_DIM(argv[1], 0) ||
        PyArray_DIM(argv[0], 1) != PyArray_DIM(argv[1], 1)) {
      Py_RETURN_FALSE;
    }

    for (int i=0; i<int(PyArray_DIM(argv[0], 0)); ++i) {
      for (int j=0; j<int(PyArray_DIM(argv[0], 1)); ++j) {
        Real* x = (Real*)(PyArray_BYTES(argv[0]) + i*PyArray_STRIDE(argv[0], 0) + j*PyArray_STRIDE(argv[0], 1));
        Real* y = (Real*)(PyArray_BYTES(argv[1]) + i*PyArray_STRIDE(argv[1], 0) + j*PyArray_STRIDE(argv[1], 1));
        Real diff = 0;
        if (*y == 0) diff = abs(*x);
        else if (*x == 0) diff = abs(*y);
        else {
          diff = abs(*y - *x)/abs(*y);
        }
        if (diff > precision) Py_RETURN_FALSE;
      }
    }
  }

  Py_RETURN_TRUE;
}


static PyObject*
postProcessTicks(PyObject* self, PyObject* args) {
  std::vector<PyObject*> argsV = unpack(args);
  if (argsV.size() == 3) {
    if (!PyArray_Check(argsV[0]) ||
        !PyArray_Check(argsV[1]) ||
        !PyFloat_Check(argsV[2])) {
      PyErr_SetString(PyExc_TypeError, (char*)" requires argument types:numpy array, numpy array, float");
      return NULL;
    }

    vector<Real>* ticks = reinterpret_cast<vector<Real>*>(VectorReal::fromPythonRef(argsV[0]));
    vector<Real>* ticksAmp = reinterpret_cast<vector<Real>*>(VectorReal::fromPythonRef(argsV[1]));
    Real period = PyFloat_AS_DOUBLE(argsV[2]);
    vector<Real> prunedTicks = essentia::postProcessTicks(*ticks, *ticksAmp, period);
    RogueVector<Real>* result = new RogueVector<Real>(prunedTicks.size(), 0.);
    for (int i=0; i<int(prunedTicks.size()); ++i) { (*result)[i] = prunedTicks[i]; }

    return VectorReal::toPythonRef(result);
  }
  else if (argsV.size() == 1) {// use the original postprocessticks from essentia 1.0
    if (!PyArray_Check(argsV[0])) {
      PyErr_SetString(PyExc_TypeError, (char*)" requires argument types: numpy array");
      return NULL;
    }

    vector<Real>* ticks = reinterpret_cast<vector<Real>*>(VectorReal::fromPythonRef(argsV[0]));
    vector<Real> prunedTicks = essentia::postProcessTicks(*ticks);
    RogueVector<Real>* result = new RogueVector<Real>(prunedTicks.size(), 0.);
    for (int i=0; i<int(prunedTicks.size()); ++i) { (*result)[i] = prunedTicks[i]; }

    return VectorReal::toPythonRef(result);
  }
  else {
    PyErr_SetString(PyExc_ValueError, "postProcessTicks requires 1 or 3 data arguments (ticks, ticksAmplitude, preferredPeriod)");
    return NULL;
  }
}



static PyMethodDef Essentia__Methods[] = {
  { "debugLevel",      (PyCFunction)debug_level,       METH_NOARGS,  "return the activated debugging modules." },
  { "setDebugLevel",   (PyCFunction)set_debug_level,   METH_O,       "set the activated debugging modules." },
  { "infoLevel",       (PyCFunction)info_level,        METH_NOARGS,  "return whether info messages should be displayed." },
  { "setInfoLevel",    (PyCFunction)set_info_level,    METH_O,       "set whether info messages should be displayed." },
  { "warningLevel",    (PyCFunction)warning_level,     METH_NOARGS,  "return whether warning messages should be displayed." },
  { "setWarningLevel", (PyCFunction)set_warning_level, METH_O,       "set whether warning messages should be displayed." },
  { "errorLevel",      (PyCFunction)error_level,       METH_NOARGS,  "return whether error messages should be displayed." },
  { "setErrorLevel",   (PyCFunction)set_error_level,   METH_O,       "set whether error messages should be displayed." },

  { "log_debug",       (PyCFunction)log_debug,             METH_VARARGS, "log the string to the given debugging module." },
  { "log_info",        (PyCFunction)log_info,              METH_O,       "log the string to the info stream." },
  { "log_warning",     (PyCFunction)log_warning,           METH_O,       "log the string to the warning stream." },
  { "log_error",       (PyCFunction)log_error,             METH_O,       "log the string to the error stream." },

  { "normalize",    normalize,      METH_O,     "returns the normalized array." },
  { "derivative",   derivative,     METH_O,     "returns the derivative of an array." },
  { "isSilent",     is_silent,      METH_O, "returns true if the frame is silent." },
  { "instantPower", instant_power,  METH_O, "returns the instant power of a frame." },
  { "nextPowerTwo", next_power_two, METH_O, "returns the next power of two." },
  { "isPowerTwo",   is_power_two,   METH_O, "returns true if argument is a power of two." },
  { "bark2hz",      barkToHz,       METH_O, "Converts a bark band to frequency in Hz" },
  { "hz2bark",      hzToBark,       METH_O, "Converts a frequency in Hz to a bark band" },
  { "mel2hz",       melToHz,        METH_O, "Converts a mel band to frequency in Hz" },
  { "hz2mel",       hzToMel,        METH_O, "Converts a frequency in Hz to a mel band" },
  { "lin2db",       linToDb,        METH_O, "Converts a linear measure of power to a measure in dB" },
  { "db2lin",       dbToLin,        METH_O, "Converts a dB measure of power to a linear measure" },
  { "db2pow",       dbToPow,        METH_O, "Converts a dB measure of power to a linear measure" },
  { "pow2db",       powToDb,        METH_O, "Converts a linear measure of power to a measure in dB" },
  { "db2amp",       dbToAmp,        METH_O, "Converts a dB measure of amplitude to a linear measure" },
  { "amp2db",       ampToDb,        METH_O, "Converts a linear measure of amplitude to a measure in dB" },
  { "info",         standard_info,  METH_VARARGS, "returns all the information about a given classic algorithm." },
  { "sinfo",        streaming_info, METH_VARARGS, "returns all the information about a given streaming algorithm." },

  { "totalProduced",   (PyCFunction)totalProduced,       METH_VARARGS, "returns the number of tokens written by algorithm's source." },
  { "connect",         (PyCFunction)connect,             METH_VARARGS, "Connects an algorithm's source to another algorithm's sink." },
  { "poolConnect",     (PyCFunction)poolConnect,         METH_VARARGS, "Connects an algorithm's source to a pool under a key name." },
  { "fileOutputConnect", (PyCFunction)fileOutputConnect, METH_VARARGS, "Connects an algorithm's source to a FileOutput." },
  { "nowhereConnect",  (PyCFunction)nowhereConnect,      METH_VARARGS, "Connects an algorithm's source to nothing." },
  { "disconnect",      (PyCFunction)disconnect,          METH_VARARGS, "Disconnects an algorithm's source from another algorithm's sink." },
  { "poolDisconnect",  (PyCFunction)poolDisconnect,      METH_VARARGS, "Disconnects an algorithm's source from a pool under a key name." },
  { "fileOutputDisconnect",  (PyCFunction)fileOutputDisconnect, METH_VARARGS, "Disconnects an algorithm's source from a FileOutput." },
  { "nowhereDisconnect", (PyCFunction)nowhereDisconnect, METH_VARARGS, "Disconnects an algorithm's source from nothing." },
  { "run",          (PyCFunction)run,                    METH_O, "Runs the given algorithm." },
  { "reset",        (PyCFunction)reset,                  METH_O, "Resets the given generator's network." },
  { "keys",         (PyCFunction)keys,                   METH_NOARGS, "returns algorithm names" },
  { "skeys",        (PyCFunction)skeys,                  METH_NOARGS, "returns streaming algorithm names" },
  { "version",      (PyCFunction)get_version,            METH_NOARGS, "returns essentia's version number" },
  { "almostEqualArray", (PyCFunction)almostEqualArray,   METH_VARARGS, "Returns true if two numpy arrays are within a given precision of each other" },
  { "postProcessTicks", (PyCFunction)postProcessTicks,   METH_VARARGS, "Purges ticks array based on ticks amplitude and the preferred period" },
  { NULL } // Sentinel
};
