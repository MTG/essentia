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

  // const char* are required for the tensorflow c API  
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


 public:
  TensorflowPredict() {
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
    declareParameter("graphFilename", "the name of the file from which to read the Tensorflow graph", "", "/home/pablo/base_model.pb");
    
    const char* inputNames[] = {"input_1"};
    const char* outputNames[] = {"output_node0"};

    std::vector<std::string> inputNamesVector = arrayToVector<std::string>(inputNames);
    std::vector<std::string> outputNamesVector = arrayToVector<std::string>(outputNames);

    declareParameter("inputs", "will look for this namespaces in poolIn. Should match the names of the input nodes in the Tensorflow graph", "", inputNamesVector);
    declareParameter("outputs", "will save the tensors on the graph nodes named after `outputs` to the same namespaces in the output pool", "", outputNamesVector);
  }

  void configure();
  void compute();
  TF_Tensor* arrayNDToTensor(const boost::const_multi_array_ref<Real, 3>& arrayND);
  boost::const_multi_array_ref<Real, 3> tensorToArrayND(const TF_Tensor* tensor, TF_Output node);
  TF_Output graphOperationByName(const char* nodeName, int index=0);

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
  }
};

} //namespace standard
} //namespace essentia

#endif // ESSENTIA_TENSORFLOWPREDICT_H
