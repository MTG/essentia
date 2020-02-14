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

#ifndef ESSENTIA_TENSORFLOWPREDICT_H
#define ESSENTIA_TENSORFLOWPREDICT_H

#include "algorithm.h"
#include "pool.h"
#include <tensorflow/c/c_api.h>


namespace essentia {
namespace standard {

class TensorflowPredict : public Algorithm {

 protected:
  Input<Pool> _poolIn;
  Output<Pool> _poolOut;

  std::string _graphFilename;
  std::vector<std::string> _inputNames;
  std::vector<std::string> _outputNames;

  std::vector<TF_Tensor*> _inputTensors;
  std::vector<TF_Tensor*> _outputTensors;

  std::vector<TF_Output> _inputNodes;
  std::vector<TF_Output> _outputNodes;

  size_t _nInputs;
  size_t _nOutputs;

  TF_Graph* _graph;
  TF_Status* _status;
  TF_ImportGraphDefOptions* _options;
  TF_SessionOptions* _sessionOptions;
  TF_Session* _session;

  bool _isTraining;
  bool _isTrainingSet;
  std::string _isTrainingName;

  bool _squeeze;

  void openGraph();
  TF_Tensor* TensorToTF(const Tensor<Real>& tensorIn);
  const Tensor<Real> TFToTensor(const TF_Tensor* tensor, TF_Output node);
  TF_Output graphOperationByName(const char* nodeName, int index=0);
  std::vector<std::string> nodeNames();

  inline std::string availableNodesInfo() {
    std::vector<std::string> nodes = nodeNames();
    std::string info = "TensorflowPredict: Available node names are:\n";
    for (std::vector<std::string>::const_iterator i = nodes.begin(); i != nodes.end() - 1; ++i) info += *i + ", ";
    return info + nodes.back() + ".\n\nReconfigure this algorithm with valid node names as inputs and outputs before starting the processing.";
  }

 public:
  TensorflowPredict() : _graph(TF_NewGraph()), _status(TF_NewStatus()),
      _options(TF_NewImportGraphDefOptions()), _sessionOptions(TF_NewSessionOptions()),
      _session(TF_NewSession(_graph, _sessionOptions, _status)) {
    declareInput(_poolIn, "poolIn", "the pool where to get the feature tensors");
    declareOutput(_poolOut, "poolOut", "the pool where to store the output tensors");
  }

  ~TensorflowPredict(){
    TF_CloseSession(_session, _status);
    TF_DeleteSessionOptions(_sessionOptions);
    TF_DeleteSession(_session, _status);
    TF_DeleteImportGraphDefOptions(_options);
    TF_DeleteStatus(_status);
    TF_DeleteGraph(_graph);
  }

  void declareParameters() {
    declareParameter("graphFilename", "the name of the file from which to read the Tensorflow graph", "", Parameter::STRING);
    declareParameter("inputs", "will look for these namespaces in poolIn. Should match the names of the input nodes in the Tensorflow graph", "", Parameter::VECTOR_STRING);
    declareParameter("outputs", "will save the tensors on the graph nodes named after `outputs` to the same namespaces in the output pool. Set the first element of this list as an empty array to print all the available nodes in the graph", "", Parameter::VECTOR_STRING);
    declareParameter("isTraining", "run the model in training mode (normalized with statistics of the current batch) instead of inference mode (normalized with moving statistics). This only applies to some models", "{true,false}", false);
    declareParameter("isTrainingName", "the name of an additional input node indicating whether the model is to be run in a training mode (for models with a training mode, leave it empty otherwise)", "", "");
    declareParameter("squeeze", "remove singleton dimensions of the inputs tensors. Does not apply to the batch dimension", "{true,false}", true);
  }

  void configure();
  void compute();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;
};

} //namespace standard
} //namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class TensorflowPredict : public StreamingAlgorithmWrapper {

 protected:
  Sink<Pool> _poolIn;
  Source<Pool> _poolOut;

 public:
  TensorflowPredict() {
    declareAlgorithm("TensorflowPredict");
    declareInput(_poolIn, TOKEN, "poolIn");
    declareOutput(_poolOut, TOKEN, "poolOut");
    _poolOut.setBufferType(BufferUsage::forSingleFrames);
  }
};

} //namespace standard
} //namespace essentia

#endif // ESSENTIA_TENSORFLOWPREDICT_H
