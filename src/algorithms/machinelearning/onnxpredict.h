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

#ifndef ESSENTIA_ONNXPREDICT_H
#define ESSENTIA_ONNXPREDICT_H

#include "algorithm.h"
#include "pool.h"

#define ORT_ENABLE_EXTENDED_API
#include <onnxruntime_cxx_api.h>

#include <unordered_set>

namespace essentia {
namespace standard {

struct TensorInfo {
    std::string name;
    ONNXTensorElementDataType type;
    std::vector<int64_t> shape;
};

class OnnxPredict : public Algorithm {

 protected:
    
  Input<Pool> _poolIn;
  Output<Pool> _poolOut;

  std::string _graphFilename;
  std::vector<std::string> _inputs;
  std::vector<std::string> _outputs;
  
  bool _squeeze;
  bool _isConfigured;

  size_t _nInputs;
  size_t _nOutputs;

  Ort::Value _inputTensor{nullptr};
  Ort::Value _outputTensor{nullptr};
    
  std::vector<Ort::Value> input_tensors;
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;
    
  Ort::Env _env{nullptr};
  Ort::SessionOptions _sessionOptions{nullptr};
  Ort::Session _session{nullptr};
    
  Ort::RunOptions _runOptions;
  Ort::AllocatorWithDefaultOptions _allocator;
  Ort::MemoryInfo _memoryInfo{ nullptr };     // Used to allocate memory for input
  Ort::Model _model;
    
  std::vector<TensorInfo> all_input_infos;
  std::vector<TensorInfo> all_output_infos;
    
  std::vector<TensorInfo> _inputNodes;
  std::vector<TensorInfo> _outputNodes;
    
  std::vector<std::string> inputNames();
  std::vector<std::string> outputNames();
  std::vector<TensorInfo> setTensorInfos(const Ort::Session&, Ort::AllocatorWithDefaultOptions&, bool);
  void printTensorInfos(const std::vector<TensorInfo>&, const std::string&);
  std::string getTensorInfos(const std::vector<TensorInfo>&, const std::string&);
  void checkName(const std::string, std::vector<TensorInfo>);
  std::string onnxTypeToString(ONNXTensorElementDataType);

  inline std::string availableInputInfo() {
    std::vector<std::string> inputs = inputNames();
    std::string info = "OnnxPredict: Available input names are:\n";
    for (std::vector<std::string>::const_iterator i = inputs.begin(); i != inputs.end() - 1; ++i) info += *i + ", ";
    return info + inputs.back() + ".\n\nReconfigure this algorithm with valid node names as inputs and outputs before starting the processing.";
  }

 public:
    
  OnnxPredict() : _env(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test")),
    _sessionOptions(Ort::SessionOptions()), _session(Ort::Session(nullptr)), _runOptions(NULL), _isConfigured(false) , _model(Ort::Model(nullptr)){
    declareInput(_poolIn, "poolIn", "the pool where to get the feature tensors");
    declareOutput(_poolOut, "poolOut", "the pool where to store the output tensors");
  }

  ~OnnxPredict(){
    all_input_infos.clear();
    all_output_infos.clear();
    _inputNodes.clear();
    _outputNodes.clear();
    input_tensors.clear();
    input_names.clear();
    output_names.clear();
  }

  void declareParameters() {
    const char* defaultTagsC[] = { "serve" };
    std::vector<std::string> defaultTags = arrayToVector<std::string>(defaultTagsC);

    declareParameter("graphFilename", "the name of the file from which to load the ONNX model", "", "");
    declareParameter("inputs", "will look for these namespaces in poolIn. Should match the names of the inputs in the ONNX model", "", Parameter::VECTOR_STRING);
    declareParameter("outputs", "will save the tensors on the model outputs named after `outputs` to the same namespaces in the output pool. Set the first element of this list as an empty array to print all the available model outputs", "", Parameter::VECTOR_STRING);
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

class OnnxPredict : public StreamingAlgorithmWrapper {

 protected:
  Sink<Pool> _poolIn;
  Source<Pool> _poolOut;

 public:
  OnnxPredict() {
    declareAlgorithm("OnnxPredict");
    declareInput(_poolIn, TOKEN, "poolIn");
    declareOutput(_poolOut, TOKEN, "poolOut");
    _poolOut.setBufferType(BufferUsage::forSingleFrames);
  }
};

} //namespace standard
} //namespace essentia

#endif // ESSENTIA_ONNXPREDICT_H
