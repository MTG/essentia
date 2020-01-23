/*
 * Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
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

using namespace std;
using namespace essentia;
using namespace standard;

const char* TensorflowPredict::name = "TensorflowPredict";
const char* TensorflowPredict::category = "Machine Learning";
const char* TensorflowPredict::description = DOC("This algorithm runs a Tensorflow graph and stores the desired tensors in a pool.\n"
"The Tensorflow graph should be stored in Protocol Buffer [1] (.pb) binary format, and should contain both the architecture and the weights of the model.\n"
"The parameter `inputs` should contain a list with the names of the input nodes that feed the model. The input Pool should contain the tensors corresponding to each input node stored using Essetia tensors."
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
  _inputNames = parameter("inputs").toVectorString();
  _outputNames = parameter("outputs").toVectorString();
  _isTraining = parameter("isTraining").toBool();
  _isTrainingName = parameter("isTrainingName").toString();
  _squeeze = parameter("squeeze").toBool();

  (_isTrainingName == "") ? _isTrainingSet = false : _isTrainingSet = true;

  _nInputs = _inputNames.size();
  _nOutputs = _outputNames.size();

  _status = TF_NewStatus();
  _graph = TF_NewGraph();
  _options = TF_NewImportGraphDefOptions();
  _sessionOptions = TF_NewSessionOptions();
  _session = TF_NewSession(_graph, _sessionOptions, _status);

  //if no file has been specified, do not do anything else
  if (!parameter("graphFilename").isConfigured()) return;

  _graphFilename = parameter("graphFilename").toString();

  openGraph();
}


void TensorflowPredict::openGraph() {
  if (!parameter("graphFilename").isConfigured()) {
    throw EssentiaException("TensorflowPredict: `graphFilename` parameter should be configured.");
  }


  // First we load and initialize the model.
  const auto f = fopen(_graphFilename.c_str(), "rb");
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

  TF_GraphImportGraphDef(_graph, buffer, _options, _status);

  TF_DeleteBuffer(buffer);

  if (TF_GetCode(_status) != TF_OK) {
    throw EssentiaException("TensorflowPredict: Error importing graph. ", TF_Message(_status));
  }

  _session = TF_NewSession(_graph, _sessionOptions, _status);

  if (TF_GetCode(_status) != TF_OK) {
    throw EssentiaException("TensorflowPredict: Error creating new session. ", TF_Message(_status));
  }
}

void TensorflowPredict::reset() {
  if (!parameter("graphFilename").isConfigured()) return;

  TF_CloseSession(_session, _status);
  if (TF_GetCode(_status) != TF_OK) {
    throw EssentiaException("TensorflowPredict: Error reseting session. ", TF_Message(_status));
  }
  
  _session = TF_NewSession(_graph, _sessionOptions, _status);
  if (TF_GetCode(_status) != TF_OK) {
    throw EssentiaException("TensorflowPredict: Error reseting session. ", TF_Message(_status));
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
    const Tensor<Real>& inputData =
        poolIn.value<Tensor<Real> >(_inputNames[i]);
    _inputTensors[i] = TensorToTF(inputData);
    _inputNodes[i] = graphOperationByName(_inputNames[i].c_str(), 0);
  }

  // Add isTraining flag if needed
  if (_isTrainingSet) {
    const int64_t dims[1] = {};
    TF_Tensor *isTraining = TF_AllocateTensor(TF_BOOL, dims, 0, 1);
    void* isTrainingValue = TF_TensorData(isTraining);

    if (isTrainingValue == nullptr) {
      TF_DeleteTensor(isTraining);
      throw EssentiaException("Error generating traning phase flag");
    }

    memcpy(isTrainingValue, &_isTraining, sizeof(bool));

    _inputTensors.push_back(isTraining);
    _inputNodes.push_back(graphOperationByName(_isTrainingName.c_str(), 0));
  }

  // Initialize output tensors.
  for (size_t i = 0; i < _nOutputs; i++) {
    _outputTensors[i] = nullptr;
    _outputNodes[i] = graphOperationByName(_outputNames[i].c_str(), 0);
  }

  // Run the Tensorflow session.
  TF_SessionRun(_session,
                nullptr,                         // Run options.
                &_inputNodes[0],                 // Input node names.
                &_inputTensors[0],               // input tensor values.
                _nInputs + (int)_isTrainingSet,  // Number of inputs.
                &_outputNodes[0],                // Output node names.
                &_outputTensors[0],              // Output tensor values.
                _nOutputs,                       // Number of outputs.
                nullptr, 0,                      // Target operations, number of targets.
                nullptr,                         // Run metadata.
                _status                          // Output status. 
               );

  if (TF_GetCode(_status) != TF_OK) {
    throw EssentiaException("TensorflowPredict: Error running the Tensorflow session. ", TF_Message(_status));
  }

  // Copy the desired tensors into the output pool.
  for (size_t i = 0; i < _nOutputs; i++) {
    poolOut.set(_outputNames[i], TFToTensor(_outputTensors[i], _outputNodes[i]));
  }

  // Deallocate tensors.
  for (size_t i = 0; i < _nInputs + (int)_isTrainingSet; i++) {
    TF_DeleteTensor(_inputTensors[i]);
  }

  for (size_t i = 0; i < _nOutputs; i++) {
    TF_DeleteTensor(_outputTensors[i]);
  }
}


TF_Tensor* TensorflowPredict::TensorToTF(
    const Tensor<Real>& tensorIn) {
  int dims = 1;
  vector<int64_t> shape;

  // Batch dimensions is the only one allowed to be singleton
  shape.push_back((int64_t)tensorIn.dimension(0));

  if (_squeeze) {
    for(int i = 1; i < tensorIn.rank(); i++) {
      if (tensorIn.dimension(i) > 1) {
        shape.push_back((int64_t)tensorIn.dimension(i));
        dims++;
      }
    }
  } else {
    dims = tensorIn.rank();
    for(int i = 1; i < dims; i++) {
        shape.push_back((int64_t)tensorIn.dimension(i));
      }
  }

  TF_Tensor* tensorOut = TF_AllocateTensor(
      TF_FLOAT, &shape[0], dims,
      (size_t)tensorIn.size() * sizeof(Real));

  if (tensorOut == nullptr) {
    throw EssentiaException("TensorflowPredict: Error generating input tensor.");
  }

  // Get a pointer to the data and fill the tensor.
  void* tensorData = TF_TensorData(tensorOut);

  if (tensorData == nullptr) {
    TF_DeleteTensor(tensorOut);
    throw EssentiaException("TensorflowPredict: Error generating input tensors data.");
  }

  memcpy(tensorData, tensorIn.data(),
         std::min(tensorIn.size() * sizeof(Real),
                  TF_TensorByteSize(tensorOut)));

  return tensorOut;
}


const Tensor<Real> TensorflowPredict::TFToTensor(
    const TF_Tensor* tensor, TF_Output node) {
  const Real* outputData = static_cast<Real*>(TF_TensorData(tensor));

  // Get the output tensor's shape.
  size_t outNDims = TF_GraphGetTensorNumDims(_graph, node, _status);

  if (TF_GetCode(_status) != TF_OK) {
    throw EssentiaException("TensorflowPredict: Error getting the output tensor's shape. ", TF_Message(_status));
  }

  // Create and array to store the shape of the tensor.
  array<long int, 4> shape {1, 1, 1, 1};
  shape[0] = (int)TF_Dim(tensor, 0);

  // We are assuming one of the following cases:
  //       1 - outNDims = 2 -> Batch + Feats
  //       2 - outNDims = 3 -> Batch + Timestamps + Feats
  //       3 - outNDims = 4 -> Batch + Channels + Timestamps + Feats
  size_t idx = 1;
  for (size_t i = shape.size() - outNDims + 1; i < shape.size(); i++, idx++) {
    shape[i] = (int)TF_Dim(tensor, idx);
  }

  // Return a const reference to the data in Eigen format.
  return TensorMap<const Real>(outputData, shape);
}


TF_Output TensorflowPredict::graphOperationByName(const char* nodeName,
                                                  int index) {
  // I don't understand the fuction of index here.
  TF_Output output = {TF_GraphOperationByName(_graph, nodeName), index};

  if (output.oper == nullptr) {
    throw EssentiaException("TensorflowPredict: Can't init node names.");
  }

  return output;
}
