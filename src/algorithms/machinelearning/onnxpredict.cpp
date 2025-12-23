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

#include "onnxpredict.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* OnnxPredict::name = "OnnxPredict";
const char* OnnxPredict::category = "Machine Learning";

const char* OnnxPredict::description = DOC("This algorithm runs a Onnx model and stores the desired output tensors in a pool.\n"
"The Onnx model should be saved in Open Neural Network Exchange (.onnx) binary format [1], and should contain both the architecture and the weights of the model.\n"
"The parameter `inputs` should contain a list with the names of the input nodes that feed the model. The input Pool should contain the tensors corresponding to each input node stored using Essentia tensors. "
"The pool namespace for each input tensor has to match the input node's name.\n"
"In the same way, the `outputs` parameter should contain the names of the tensors to save. These tensors will be stored inside the output pool under a namespace that matches the tensor's name. "
"To print a list with all the available nodes in the graph set the first element of `outputs` as an empty string (i.e., \"\")."
"\n"
"This algorithm is a wrapper for the ONNX Runtime Inferencing API [2]. The first time it is configured with a non-empty `graphFilename` it will try to load the contained graph and to attach a ONNX session to it. "
"The reset method deletes the model inputs and outputs internally stored in a vector. "
"By reconfiguring the algorithm the graph is reloaded and the reset method is called.\n"
"\n"
"References:\n"
"  [1] ONNX - The open standard for machine learning interoperability.\n"
"  https://onnx.ai/onnx/intro/\n\n"
"  [2] ONNX Runtime API - a cross-platform machine-learning model accelerator, with a flexible interface to integrate hardware-specific libraries.\n"
"  https://onnxruntime.ai/docs/");


void OnnxPredict::configure() {
  _graphFilename = parameter("graphFilename").toString();
  _deviceId = parameter("deviceId").toInt();
  std::string opt = parameter("optimizationLevel").toString();

  if (opt == "disable_all") {
    _optimizationLevel = OnnxOptimizationLevel::DISABLE_ALL;
  }
  else if (opt == "basic") {
    _optimizationLevel = OnnxOptimizationLevel::BASIC;
  }
  else if (opt == "extended") {
    _optimizationLevel = OnnxOptimizationLevel::EXTENDED;
  }
  else if (opt == "all") {
    _optimizationLevel = OnnxOptimizationLevel::ALL;
  }
  else {
    throw EssentiaException(
      "OnnxPredict: invalid optimizationLevel: " + opt + ". Choices: {disable_all,basic,extended,all}"
    );
  }
    
  if ((_graphFilename.empty()) and (_isConfigured)) {
    E_WARNING("OnnxPredict: You are trying to update a valid configuration with invalid parameters. "
              "If you want to update the configuration specify a valid `graphFilename` parameter.");
  };

  // Do not do anything if we did not get a non-empty model name.
  if (_graphFilename.empty()) return;
    
  reset();
      
  // get input and output info (names, type and shapes)
  all_input_infos = setTensorInfos(*_session, _allocator, "inputs");
  all_output_infos = setTensorInfos(*_session, _allocator, "outputs");
    
  // read inputs and outputs as input parameter
  _inputs = parameter("inputs").toVectorString();
  _outputs = parameter("outputs").toVectorString();
    
  _squeeze = parameter("squeeze").toBool();

  _nInputs = _inputs.size();
  _nOutputs = _outputs.size();
        
  // excepts if no inputs are defined
  if (_nInputs == 0){
    throw EssentiaException("No model input was defined.\n" + availableInputInfo());
  }

  // excepts if no outputs are defined
  if (_nOutputs == 0){
    throw EssentiaException("No model output was defined.\n" + availableOutputInfo());
  }

  // If the first output name is empty just print out the list of nodes and return.
  if (_outputs[0] == "") {
    E_INFO(getTensorInfos(all_input_infos, "Model Inputs"));
    E_INFO(getTensorInfos(all_output_infos, "Model Outputs"));
    return;
  }

  _isConfigured = true;

  // check model has input and output https://github.com/microsoft/onnxruntime-inference-examples/blob/7a635daae48450ff142e5c0848a564b245f04112/c_cxx/model-explorer/model-explorer.cpp#L99C3-L100C63
  for (int i = 0; i < _inputs.size(); i++) {
    for (int j = 0; j < all_input_infos.size(); j++) {
      if (_inputs[i] == all_input_infos[j].name){
        _inputNodes.push_back(all_input_infos[j]);
      }
    }
  }

  // Check if _inputNodes is empty - release an exception instead
  if (!_inputNodes.size())
    throw EssentiaException("No input node was found.\n" + availableInputInfo());
      
  for (int i = 0; i < _outputs.size(); i++) {
    for (int j = 0; j < all_output_infos.size(); j++) {
      if (_outputs[i] == all_output_infos[j].name){
        _outputNodes.push_back(all_output_infos[j]);
      }
    }
  }
  
  // Check if _outputNodes is empty - release an exception instead
  if (!_outputNodes.size())
    throw EssentiaException("No output node was found.\n" + availableOutputInfo());
    
  for (size_t i = 0; i < _nInputs; i++) {
    checkName(_inputs[i], all_input_infos);
  }
    
  for (size_t i = 0; i < _nOutputs; i++) {
    checkName(_outputs[i], all_output_infos);
  }
}

std::vector<TensorInfo> OnnxPredict::setTensorInfos(const Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator, const std::string& port) {
    
    std::vector<TensorInfo> infos;
    
    size_t count = (port == "inputs") ? session.GetInputCount() : session.GetOutputCount();
    auto names_raw = (port == "inputs") ? session.GetInputNames() : session.GetOutputNames();

    for (size_t i = 0; i < count; ++i) {
        auto name_raw = names_raw[i];

        std::string name(name_raw);
        Ort::TypeInfo type_info = (port == "inputs") ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);

        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        TensorInfo info;
        info.name = name;
        info.type = tensor_info.GetElementType();
        info.shape = tensor_info.GetShape();

        infos.push_back(std::move(info));
    }

    return infos;
}

void OnnxPredict::printTensorInfos(const std::vector<TensorInfo>& infos, const std::string& label) {
    E_INFO("=== " << label << " ===\n");
    for (const auto& info : infos) {
        E_INFO("[Name] " << info.name);
        E_INFO("  [Type] " << info.type);
        E_INFO("  [Shape] [");
        for (size_t j = 0; j < info.shape.size(); ++j) {
            E_INFO(info.shape[j]);
            if (j + 1 < info.shape.size()) E_INFO(", ");
        }
        E_INFO("]\n");
    }
}

std::string OnnxPredict::getTensorInfos(const std::vector<TensorInfo>& infos, const std::string& label) {
  std::string out;
  out += "=== " + label + " ===\n";
  for (const auto& info : infos) {
    out += "[Name] " + info.name + "\n";
    std::string type_str = onnxTypeToString(info.type);
    out += "\t[Type] " + type_str + "\n";
    out += "\t[Shape] [";
    for (size_t j = 0; j < info.shape.size(); ++j) {
      out += info.shape[j];
      if (j + 1 < info.shape.size()) out += ", ";
    }
    out += "]\n";
  }
  return out;
}

void OnnxPredict::reset() {

  input_names.clear();
  output_names.clear();
  _inputNodes.clear();
  _outputNodes.clear();

  try{

    // Reset session
    _session.reset();
  
    // Reset SessionOptions by constructing a fresh object
    _sessionOptions = Ort::SessionOptions{};
  
    // Auto-detect EPs
    #ifdef USE_CUDA
    if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end()) {
      OrtSessionOptionsAppendExecutionProvider_CUDA(_sessionOptions, _deviceId);
      E_INFO("✅ Using CUDA Execution Provider (GPU " << _deviceId << ")");
    }
    #endif

    #ifdef USE_METAL
    if (std::find(providers.begin(), providers.end(), "MetalExecutionProvider") != providers.end()) {
      OrtSessionOptionsAppendExecutionProvider_Metal(_sessionOptions, _deviceId);
      E_INFO("✅ Using Metal Execution Provider (GPU " << _deviceId << ")");
    }
    #endif

    #ifdef USE_COREML
    if (std::find(providers.begin(), providers.end(), "CoreMLExecutionProvider") != providers.end()) {
      OrtSessionOptionsAppendExecutionProvider_CoreML(_sessionOptions, _deviceId);
      E_INFO("✅ Using Core ML Execution Provider (GPU " << _deviceId << ")");
    }
    #endif
        
    // Set graph optimization level - Map our enum to ONNX Runtime | Check https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
    switch (_optimizationLevel) {
      case OnnxOptimizationLevel::DISABLE_ALL:
        _sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        break;
      case OnnxOptimizationLevel::BASIC:
        _sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        break;
      case OnnxOptimizationLevel::EXTENDED:
        _sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        break;
      case OnnxOptimizationLevel::ALL:
        _sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        break;
    }
    _sessionOptions.SetIntraOpNumThreads(0);
      
    // Initialize session
    _session = std::make_unique<Ort::Session>(_env, _graphFilename.c_str(), _sessionOptions);

  }
  catch (Ort::Exception e) {
    // Fallback only if optimization > BASIC
    if (_optimizationLevel != OnnxOptimizationLevel::BASIC &&
        _optimizationLevel != OnnxOptimizationLevel::DISABLE_ALL) {
      E_WARNING(
                "OnnxPredict: graph optimization level failed ("
                + std::string(e.what())
                + "), retrying with BASIC optimization"
                );
      // Fallback to BASIC
      _sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
      _session = std::make_unique<Ort::Session>(_env, _graphFilename.c_str(), _sessionOptions);
    }
    else
      // No fallback possible
      throw EssentiaException("OnnxPredict: session creation failed: " + std::string(e.what()), e.GetOrtErrorCode());
  }
  
  E_INFO("OnnxPredict: Successfully loaded graph file: `" << _graphFilename << "`");
}

void OnnxPredict::compute() {
    
  if (!_isConfigured) {
    throw EssentiaException("OnnxPredict: This algorithm is not configured. To configure this algorithm you "
                            "should specify a valid `graphFilename`, `inputs` and `outputs` as input parameter.");
  }

  const Pool& poolIn = _poolIn.get();
  Pool& poolOut = _poolOut.get();

  std::vector<std::vector<float>> inputDataVector;  // <-- keeps inputs alive
  std:vector<std::vector<int64_t>> shapes;      // <-- keeps shapes alive
  
  if (!input_tensors.empty())
      input_tensors.clear();                        // <-- destroy input tensors

    
  // Parse the input tensors from the pool into ONNX Runtime tensors.
  for (size_t i = 0; i < _nInputs; i++) {
    
    const Tensor<Real>& inputData = poolIn.value<Tensor<Real> >(_inputs[i]);
    
    // Step 1: Get tensor shape
    std::vector<int64_t> shape;
    int dims = 1;

    shape.push_back((int64_t)inputData.dimension(0));
    
    if (_squeeze) {
    
      for(int j = 1; j < inputData.rank(); j++) {
        if (inputData.dimension(j) > 1) {
          shape.push_back((int64_t)inputData.dimension(j));
          dims++;
        }
      }
          
      // There should be at least 2 dimensions (batch, data)
      if (dims == 1) {
        shape.push_back((int64_t) 1);
        dims++;
      }
        
    } else {
      dims = inputData.rank();
      for(int j = 1; j < dims; j++) {
        shape.push_back((int64_t)inputData.dimension(j));
      }
    }
             
    // Step 2: keep Real (float32) as-is ---
    inputDataVector.emplace_back(inputData.size());
    // Essentia::Real is already float32 by default, so no need to cast.
    // We copy directly into the input vector that will be fed to ONNX tensor.
    std::copy(inputData.data(), inputData.data() + inputData.size(), inputDataVector.back().begin());
    
    // Step 3: Create ONNX Runtime tensor
    #ifdef USE_CUDA
    if (_sessionOptions.GetExecutionProviderCount() > 0 &&
        std::string(_sessionOptions.GetExecutionProviderName(0)) == "CUDAExecutionProvider") {
        _memoryInfo = Ort::MemoryInfo::CreateCuda(_deviceId, OrtMemTypeDefault);
    } else
    #endif
    {
        _memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    }

    if (_memoryInfo == nullptr) {
        throw EssentiaException("OnnxPredict: Error allocating memory for input tensor.");
    }
      
    if (_memoryInfo == NULL) {
      throw EssentiaException("OnnxRuntimePredict: Error allocating memory for input tensor.");
    }
    
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(_memoryInfo, inputDataVector.back().data(), inputDataVector.back().size(), shape.data(), shape.size()));
    shapes.push_back(shape);
  }

  // Define input and output names
  for (const auto& tensorInfo : _inputNodes) {
    input_names.push_back(tensorInfo.name.c_str());
  }
    
  for (const auto& tensorInfo : _outputNodes) {
    output_names.push_back(tensorInfo.name.c_str());
  }
    
  // Run the Onnxruntime session.
  auto output_tensors = _session->Run(_runOptions,                       // Run options.
                                      input_names.data(),                // Input node names.
                                      input_tensors.data(),              // Input tensor values.
                                      _nInputs,                          // Number of inputs.
                                      output_names.data(),               // Output node names.
                                      _nOutputs                          // Number of outputs.
                                      );

  // Map output tensors to pool
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    
    const Real* outputData = output_tensors[i].GetTensorData<Real>();
    
    // Create and array to store the output tensor shape.
    array<long int, 4> _shape {1, 1, 1, 1};
    _shape[0] = (int)shapes[0][0];
    
    for (size_t j = 1; j < _outputNodes[i].shape.size(); j++){
      int shape_idx = _shape.size() - j;
      _shape[shape_idx] = (int)_outputNodes[i].shape[_outputNodes[i].shape.size() - j];
    }
    
    // Store tensor in pool
    const Tensor<Real> tensorMap = TensorMap<const Real>(outputData, _shape);
    poolOut.set(_outputs[i], tensorMap);
  }
    
}


void OnnxPredict::checkName(const string nodeName, std::vector<TensorInfo> _infos) {

  vector<string> _names;
    
  for(int i = 0; i< _infos.size(); i++) {
      _names.push_back(_infos[i].name);
  }

  std::unordered_set<std::string> lookup(_names.begin(), _names.end());
  if (lookup.find(nodeName) == lookup.end())
    throw EssentiaException("OnnxPredict: `" + nodeName + "` is not a valid input node name. Make sure that all "
                            "your inputs are defined in the node list.");
}


vector<string> OnnxPredict::inputNames() {
  
  vector<string> inputNames;
  
  // inputs
  for(int i = 0; i< all_input_infos.size(); i++) {
     inputNames.push_back(all_input_infos[i].name);
  }

  return inputNames;
}

vector<string> OnnxPredict::outputNames() {
  
  vector<string> outputNames;
  
  // inputs
  for(int i = 0; i< all_input_infos.size(); i++) {
     outputNames.push_back(all_input_infos[i].name);
  }

  return outputNames;
}

std::string OnnxPredict::onnxTypeToString(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return "uint16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return "string";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return "bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: return "uint32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: return "uint64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: return "complex64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: return "complex128";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return "bfloat16";
        default: return "unknown";
    }
}
