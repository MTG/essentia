/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#include "tensorflowpredict.h"
#include "algorithmfactory.h"
#include "tnt/tnt2vector.h"
#include "boost/multi_array.hpp"

using namespace std;
using namespace essentia;
using namespace standard;
using namespace boost;

const char* TensorflowPredict::name = "TensorflowPredict";
const char* TensorflowPredict::category = "Machine Learning";
const char* TensorflowPredict::description = DOC("This algorithm runs a Tensorflow graph and stores the desired tensors in a pool.\n"
"The Tensorflow graph should be stored in Protocol Buffer [1] (.pb) binary format, and should contain both the architecture and the weights of the model.\n"
"The parameter `inputs` should contain a list with the names of the input nodes that feed the model. The input Pool should contain the tensors corresponding to each input node stored using EssetiaArrayND. "
"The pool namespace for each input tensor has to match the input node's name.\n"
"In the same way, the `outputs` parameter should contain the names of the nodes whose tensors are desired to save. These tensors will be stored inside the output pool under a namespace that matches the name of the node.\n"
"\n"
"Note: This algorithm is a wrapper for the Tensorflow C API[2]."
"\n"
"References:\n"
"  [1] TensorFlow - An open source machine learning library for research and production.\n"
"  https://www.tensorflow.org/extend/tool_developers/#protocol_buffers\n"
"  [2] TensorFlow - An open source machine learning library for research and production.\n"
"  https://www.tensorflow.org/api_docs/cc/");


static void DeallocateBuffer(void* data, size_t) {
  free(data);
}

void TensorflowPredict::configure() {
  string garphFilename = parameter("graphFilename").toString();
  _inputNames = parameter("inputs").toVectorString();
  _outputNames = parameter("outputs").toVectorString();

  _nInputs = _inputNames.size();
  _nOutputs = _outputNames.size();

  // First we load and initialize the model.
  const auto f = fopen(garphFilename.c_str(), "rb");
  if (f == nullptr) {
    throw EssentiaException(
        "TensorflowPredict: could not open the tensorflow graph file.");
  }

  fseek(f, 0, SEEK_END);
  const auto fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  // Graph size sanity check.
  if (fsize < 1) {
    fclose(f);
    throw(EssentiaException("TensorflowPredict: Graph file is empty."));
  }

  // Reserve memory and read the graph.
  const auto data = malloc(fsize);
  fread(data, fsize, 1, f);
  fclose(f);

  TF_Buffer* buffer = TF_NewBuffer();
  buffer->data = data;
  buffer->length = fsize;
  buffer->data_deallocator = DeallocateBuffer;

  _graph = TF_NewGraph();
  _status = TF_NewStatus();
  _options = TF_NewImportGraphDefOptions();

  TF_GraphImportGraphDef(_graph, buffer, _options, _status);

  TF_DeleteBuffer(buffer);

  if (TF_GetCode(_status) != TF_OK) {
    throw EssentiaException("TensorflowPredict: Graph status is ", _status);
  }

  _sessionOptions = TF_NewSessionOptions();
  _session = TF_NewSession(_graph, _sessionOptions, _status);

  if (TF_GetCode(_status) != TF_OK) {
    throw EssentiaException("TensorflowPredict: Session status is ", _status);
  }
}

void TensorflowPredict::compute() {
  const Pool& poolIn = _poolIn.get();
  Pool& poolOut = _poolOut.get();

  // Allocate input and output tensors.
  _inputTensors.resize(_nInputs);
  _inputNodes.resize(_nInputs);

  _outputTensors.resize(_nOutputs);
  _outputNodes.resize(_nOutputs);

  // Parse the input tensors from the pool into Tensorflow tensors.
  for (size_t i = 0; i < _nInputs; i++) {
    const_multi_array_ref<Real, 3> inputData(
        poolIn.value<vector<multi_array<Real, 3> > >(_inputNames[i])[0]);

    _inputTensors[i] = arrayNDToTensor(inputData);
    _inputNodes[i] = graphOperationByName(_inputNames[i].c_str(), 0);
  }

  // Initialize output tensors.
  for (size_t i = 0; i < _nOutputs; i++) {
    _outputTensors[i] = nullptr;
    _outputNodes[i] = graphOperationByName(_outputNames[i].c_str(), 0);
  }

  // Run the Tensorflow session.
  TF_SessionRun(_session,
                nullptr,             // Run options.
                &_inputNodes[0],     // Input node names.
                &_inputTensors[0],   // input tensor values.
                _nInputs,            // Number of inputs.
                &_outputNodes[0],    // Output node names.
                &_outputTensors[0],  // Output tensor values.
                _nOutputs,           // Number of outputs.
                nullptr, 0,          // Target operations, number of targets.
                nullptr,             // Run metadata.
                _status              // Output status. 
               );

  if (TF_GetCode(_status) != TF_OK) {
    TF_DeleteStatus(_status);
    throw EssentiaException("Error running the Tensorflow session");
  }

  // Copy the desired tensors into the output pool.
  for (size_t i = 0; i < _nOutputs; i++) {
    poolOut.add(_outputNames[i], multi_array<Real, 3>(tensorToArrayND(
                                     _outputTensors[i], _outputNodes[i])));
  }

  // Deallocate tensors.
  for (size_t i = 0; i < _nInputs; i++) {
    TF_DeleteTensor(_inputTensors[i]);
  }

  for (size_t i = 0; i < _nOutputs; i++) {
    TF_DeleteTensor(_outputTensors[i]);
  }
}


TF_Tensor* TensorflowPredict::arrayNDToTensor(
    const const_multi_array_ref<Real, 3>& arrayND) {
  TF_Tensor* tensor = TF_AllocateTensor(
      TF_FLOAT, (const int64_t*)arrayND.shape(), arrayND.num_dimensions(),
      arrayND.num_elements() * sizeof(Real));

  if (tensor == nullptr) {
    throw EssentiaException("Error generating input tensor.");
  }

  // Get a pointer to the data and fill the tensor.
  void* tensorData = TF_TensorData(tensor);

  if (tensorData == nullptr) {
    TF_DeleteTensor(tensor);
    throw EssentiaException("Error generating input tensors data");
  }

  // Why min?
  memcpy(tensorData, arrayND.origin(),
         std::min(arrayND.num_elements() * sizeof(Real),
                  TF_TensorByteSize(tensor)));

  return tensor;
}


const_multi_array_ref<Real, 3> TensorflowPredict::tensorToArrayND(
    const TF_Tensor* tensor, TF_Output node) {
  const Real* outputData = static_cast<Real*>(TF_TensorData(tensor));

  // Get the output tensor's shape.
  size_t outNDims = TF_GraphGetTensorNumDims(_graph, node, _status);

  if (TF_GetCode(_status) != TF_OK) {
    TF_DeleteStatus(_status);
    throw EssentiaException("Error geting the output tensor's shape.");
  }

  // Create a boost array to store the shape of the tensor.
  boost::array<multi_array<int, 3>::index, 3> shape;
  for (size_t i = 0; i < outNDims; i++) {
    shape[i] = (int)TF_Dim(tensor, i);
  }

  // Return a const reference to the data in the Boost format.
  return const_multi_array_ref<Real, 3>(outputData, shape);
}


TF_Output TensorflowPredict::graphOperationByName(const char* nodeName,
                                                  int index) {
  // I don't understand the fuction of index here.
  TF_Output output = {TF_GraphOperationByName(_graph, nodeName), index};

  if (output.oper == nullptr) {
    throw EssentiaException("Can't init node names");
  }

  return output;
}