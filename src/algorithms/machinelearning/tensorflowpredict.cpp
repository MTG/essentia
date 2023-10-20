/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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
const char* TensorflowPredict::description = DOC("This algorithm runs a Tensorflow graph and stores the desired output tensors in a pool.\n"
"The Tensorflow graph should be stored in Protocol Buffer (.pb) binary format [1] or as a SavedModel [2], and should contain both the architecture and the weights of the model.\n"
"The parameter `inputs` should contain a list with the names of the input nodes that feed the model. The input Pool should contain the tensors corresponding to each input node stored using Essetia tensors. "
"The pool namespace for each input tensor has to match the input node's name.\n"
"In the same way, the `outputs` parameter should contain the names of the tensors to save. These tensors will be stored inside the output pool under a namespace that matches the tensor's name. "
"TensorFlow nodes return tensors identified as `nodeName:n`, where `n` goes from 0 to the number of outputs of the node - 1. "
"The first output tensor of a node can be referred implicitly (e.g., `nodeName`) or explicitly (e.g., `nodeName:0`), and the subsequent tensors require the specific index (e.g., nodeName:1, nodeName:2). "
"In TensorFlow 2 SavedModels all the outputs of the model are contained in the `PartitionedCall` node (e.g., `PartitionedCall:0`, `PartitionedCall:1`). "
"You can use TenorFlow's `saved_model_cli` to find the relation of outputs and `PartitionedCall` indices, for example:\n"
"\n"
">>>    saved_model_cli show --dir SavedModel/ --all\n"
">>>    ...\n"
">>>    The given SavedModel SignatureDef contains the following output(s):\n"
">>>      outputs['activations'] tensor_info:\n"
">>>          dtype: DT_FLOAT\n"
">>>          shape: (1, 400)\n"
">>>          name: PartitionedCall:0\n"
">>>      outputs['embeddings'] tensor_info:\n"
">>>          dtype: DT_FLOAT\n"
">>>          shape: (1, 1280)\n"
">>>          name: PartitionedCall:1\n"
">>>    ...\n"
"\n"
"To print a list with all the available nodes in the graph set the first element of `outputs` as an empty string (i.e., \"\")."
"\n"
"This algorithm is a wrapper for the Tensorflow C API [3]. The first time it is configured with a non-empty `graphFilename` it will try to load the contained graph and to attach a Tensorflow session to it. "
"The reset method deletes the current session (and the resources attached to it) and creates a new one relying on the available graph. "
"By reconfiguring the algorithm the graph is reloaded and the reset method is called.\n"
"\n"
"References:\n"
"  [1] TensorFlow - An open source machine learning library for research and production.\n"
"  https://www.tensorflow.org/extend/tool_developers/#protocol_buffers\n\n"
"  [2] TensorFlow - An open source machine learning library for research and production.\n"
"  https://www.tensorflow.org/guide/saved_model\n\n"
"  [3] TensorFlow - An open source machine learning library for research and production.\n"
"  https://www.tensorflow.org/api_docs/cc/");


static void DeallocateBuffer(void* data, size_t) {
  free(data);
}


void TensorflowPredict::configure() {
  _savedModel = parameter("savedModel").toString();
  _graphFilename = parameter("graphFilename").toString();

  if ((_savedModel.empty()) and (_graphFilename.empty()) and (_isConfigured)) {
    E_WARNING("TensorflowPredict: You are trying to update a valid configuration with invalid parameters. "
              "If you want to update the configuration specify a valid `graphFilename` or `savedModel` parameter.");
  };

  // Do not do anything if we did not get a non-empty model name.
  if ((_savedModel.empty()) and (_graphFilename.empty())) return;

  _tags = parameter("tags").toVectorString();

  _inputNames = parameter("inputs").toVectorString();
  _outputNames = parameter("outputs").toVectorString();
  _isTraining = parameter("isTraining").toBool();
  _isTrainingName = parameter("isTrainingName").toString();
  _squeeze = parameter("squeeze").toBool();

  (_isTrainingName == "") ? _isTrainingSet = false : _isTrainingSet = true;

  _nInputs = _inputNames.size();
  _nOutputs = _outputNames.size();

  // Allocate input and output tensors.
  _inputTensors.resize(_nInputs + int(_isTrainingSet));
  _inputNodes.resize(_nInputs + int(_isTrainingSet));

  _outputTensors.resize(_nOutputs);
  _outputNodes.resize(_nOutputs);

  TF_DeleteGraph(_graph);
  _graph = TF_NewGraph();

  openGraph();

  _isConfigured = true;
  reset();

  // If the first output name is empty just print out the list of nodes and return.
  if (_outputNames[0] == "") {
    E_INFO(availableNodesInfo());

    return;
  }

  // Initialize the list of input and output node names.
  for (size_t i = 0; i < _nInputs; i++) {
    _inputNodes[i] = graphOperationByName(_inputNames[i]);
  }

  for (size_t i = 0; i < _nOutputs; i++) {
    _outputNodes[i] = graphOperationByName(_outputNames[i]);
  }

  // Add isTraining flag if needed.
  if (_isTrainingSet) {
    const int64_t dims[1] = {};
    TF_Tensor *isTraining = TF_AllocateTensor(TF_BOOL, dims, 0, 1);
    void* isTrainingValue = TF_TensorData(isTraining);

    if (isTrainingValue == NULL) {
      TF_DeleteTensor(isTraining);
      throw EssentiaException("TensorflowPredict: Error generating training phase flag");
    }

    memcpy(isTrainingValue, &_isTraining, sizeof(bool));

    _inputTensors[_nInputs] = isTraining;
    _inputNodes[_nInputs] = graphOperationByName(_isTrainingName);
  }
}


void TensorflowPredict::openGraph() {
  // Prioritize savedModel when both are specified.
  if (!_savedModel.empty()) {
    std::vector<char*> tags_c;
    tags_c.reserve(_tags.size());
    for (size_t i = 0; i < _tags.size(); i++) {
      tags_c.push_back(const_cast<char*>(_tags[i].c_str()));
    }

    TF_LoadSessionFromSavedModel(_sessionOptions, _runOptions,
      _savedModel.c_str(), &tags_c[0], (int)tags_c.size(),
      _graph, NULL, _status);

    if (TF_GetCode(_status) != TF_OK) {
      throw EssentiaException("TensorflowPredict: Error importing SavedModel specified in the `savedModel` parameter. ", TF_Message(_status));
    }

    E_INFO("TensorflowPredict: Successfully loaded SavedModel: `" << _savedModel << "`");

    return;
  }

  if (!_graphFilename.empty()) {
    // First we load and initialize the model.
    const auto f = fopen(_graphFilename.c_str(), "rb");
    if (f == NULL) {
      throw EssentiaException(
          "TensorflowPredict: could not open the Tensorflow graph file.");
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

    E_INFO("TensorflowPredict: Successfully loaded graph file: `" << _graphFilename << "`");
  }
}


void TensorflowPredict::reset() {
  if (!_isConfigured) return;

  TF_CloseSession(_session, _status);
  if (TF_GetCode(_status) != TF_OK) {
    throw EssentiaException("TensorflowPredict: Error closing session. ", TF_Message(_status));
  }

  TF_DeleteSession(_session, _status);
  if (TF_GetCode(_status) != TF_OK) {
    throw EssentiaException("TensorflowPredict: Error deleting session. ", TF_Message(_status));
  }

  _session = TF_NewSession(_graph, _sessionOptions, _status);
  if (TF_GetCode(_status) != TF_OK) {
    throw EssentiaException("TensorflowPredict: Error creating new session after reset. ", TF_Message(_status));
  }
}


void TensorflowPredict::compute() {
  if (!_isConfigured) {
    throw EssentiaException("TensorflowPredict: This algorithm is not configured. To configure this algorithm you "
                            "should specify a valid `graphFilename` or `savedModel` as input parameter.");
  }

  const Pool& poolIn = _poolIn.get();
  Pool& poolOut = _poolOut.get();

  // Parse the input tensors from the pool into Tensorflow tensors.
  for (size_t i = 0; i < _nInputs; i++) {
    const Tensor<Real>& inputData =
        poolIn.value<Tensor<Real> >(_inputNames[i]);
    _inputTensors[i] = TensorToTF(inputData);
  }

  // Initialize output tensors.
  for (size_t i = 0; i < _nOutputs; i++) {
    _outputTensors[i] = NULL;
  }

  // Run the Tensorflow session.
  TF_SessionRun(_session,
                NULL,                            // Run options.
                &_inputNodes[0],                 // Input node names.
                &_inputTensors[0],               // input tensor values.
                _nInputs + (int)_isTrainingSet,  // Number of inputs.
                &_outputNodes[0],                // Output node names.
                &_outputTensors[0],              // Output tensor values.
                _nOutputs,                       // Number of outputs.
                NULL, 0,                         // Target operations, number of targets.
                NULL,                            // Run metadata.
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
  for (size_t i = 0; i < _nInputs; i++) {
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

  // With squeeze, the Batch dimension is the only one allowed to be singleton
  shape.push_back((int64_t)tensorIn.dimension(0));

  if (_squeeze) {
    for(int i = 1; i < tensorIn.rank(); i++) {
      if (tensorIn.dimension(i) > 1) {
        shape.push_back((int64_t)tensorIn.dimension(i));
        dims++;
      }
    }

    // There should be at least 2 dimensions (batch, data)
    if (dims == 1) {
      shape.push_back((int64_t) 1);
      dims++;
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

  if (tensorOut == NULL) {
    throw EssentiaException("TensorflowPredict: Error generating input tensor.");
  }

  // Get a pointer to the data and fill the tensor.
  void* tensorData = TF_TensorData(tensorOut);

  if (tensorData == NULL) {
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


TF_Output TensorflowPredict::graphOperationByName(const string nodeName) {
  int index = 0;
  const char* name = nodeName.c_str();
  string newNodeName;

  // TensorFlow operations (or nodes from the graph perspective) return tensors named <nodeName:n>, where n goes
  // from 0 to the number of outputs. The first output tensor of a node can be extracted implicitly (nodeName)
  // or explicitly (nodeName:0). To extract the subsequent tensors, the index has to be specified (nodeName:1,
  // nodeName:2, ...).
  string::size_type n = nodeName.find(':');
  if (n != string::npos) {
    try {
      newNodeName = nodeName.substr(0, n);
      name = newNodeName.c_str();
      index = stoi(nodeName.substr(n + 1, nodeName.size()));

    } catch (const invalid_argument& ) {
      throw EssentiaException("TensorflowPredict: `" + nodeName + "` is not a valid node name. Make sure that all "
                              "your inputs and outputs follow the pattern `nodeName:n`, where `n` in an integer that "
                              "goes from 0 to the number of outputs of the node - 1.");
    }

  }

  TF_Operation* oper = TF_GraphOperationByName(_graph, name);

  TF_Output output = {oper, index};

  if (output.oper == NULL) {
    throw EssentiaException("TensorflowPredict: '" + string(nodeName) +
                            "' is not a valid node name of this graph.\n" +
                            availableNodesInfo());
  }

  int n_outputs = TF_OperationNumOutputs(oper);
  if (index > n_outputs - 1) {
    throw EssentiaException("TensorflowPredict: Asked for the output with index `" + to_string(index) +
                            "`, but the node `" + name + "` only has `" + to_string(n_outputs) + "` outputs.");
  }

  return output;
}

vector<string> TensorflowPredict::nodeNames() {
  size_t pos = 0;
  TF_Operation *oper;
  vector<string> nodeNames;

  while ((oper = TF_GraphNextOperation(_graph, &pos)) != NULL) {
    nodeNames.push_back(string(TF_OperationName(oper)));
  }

  return nodeNames;
}
