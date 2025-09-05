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

const char* OnnxPredict::description = DOC("This algorithm runs a Onnx graph and stores the desired output tensors in a pool.\n"
"The Onnx graph should be stored in Open Neural Network Exchange (.onnx) binary format [1], and should contain both the architecture and the weights of the model.\n"
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
    
  if ((_graphFilename.empty()) and (_isConfigured)) {
    E_WARNING("OnnxPredict: You are trying to update a valid configuration with invalid parameters. "
              "If you want to update the configuration specify a valid `graphFilename` parameter.");
  };

  // Do not do anything if we did not get a non-empty model name.
  if (_graphFilename.empty()) return;
    
  try{
    // Define environment
    _env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "multi_io_inference"); // {"default", "test", "multi_io_inference"}
      
    // Set graph optimization level - check https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
    _sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    // To enable model serialization after graph optimization set this
    _sessionOptions.SetOptimizedModelFilePath("optimized_file_path");
    _sessionOptions.SetIntraOpNumThreads(1);
        
    // Initialize session
    _session = Ort::Session(_env, _graphFilename.c_str(), _sessionOptions);
  }
  catch (Ort::Exception oe) {
    cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() <<   ".\n";
    return;
  }
      
  // get input and output info (names, type and shapes)
  all_input_infos = setTensorInfos(_session, _allocator, true);
  all_output_infos = setTensorInfos(_session, _allocator, false);
    
  // read inputs and outputs as input parameter
  _inputs = parameter("inputs").toVectorString();
  _outputs = parameter("outputs").toVectorString();
    
  _squeeze = parameter("squeeze").toBool();

  _nInputs = _inputs.size();
  _nOutputs = _outputs.size();
    
  // cout << "_inputs.size(): " << _nInputs << ".\n";
    
  // use the first input when no input is defined
  if (_nInputs == 0){
    // take the first input
    _inputs.push_back(_session.GetInputNames()[0]);
    _nInputs = _inputs.size();
    // inform the first model input will be used
    E_INFO("OnnxPredict: using the first model input '" + _inputs[0] + "'");
  }

  // define _outputs with the first model output when no output is provided
  if (_nOutputs == 0){
    // take the first output
    _outputs.push_back(_session.GetOutputNames()[0]);
    _nOutputs = _outputs.size();
    // inform the first model input will be used
    E_INFO("OnnxPredict: using the first model output '" + _outputs[0] + "'");
  }

  // If the first output name is empty just print out the list of nodes and return.
  if (_outputs[0] == "") {
    E_INFO(getTensorInfos(all_input_infos, "Model Inputs"));
    E_INFO(getTensorInfos(all_output_infos, "Model Outputs"));
    return;
  }
    
  // check model has input and output https://github.com/microsoft/onnxruntime-inference-examples/blob/7a635daae48450ff142e5c0848a564b245f04112/c_cxx/model-explorer/model-explorer.cpp#L99C3-L100C63

  for (int i = 0; i < _inputs.size(); i++) {
    for (int j = 0; j < all_input_infos.size(); j++) {
      if (_inputs[i] == all_input_infos[j].name){
        _inputNodes.push_back(all_input_infos[j]);
      }
    }
  }
  
  if (_inputNodes.size() == 0){

    std::string s;
    for (const auto &piece : _inputs) {
      s += piece;
      s += " ";
    }
      
    throw EssentiaException(availableInputInfo());
  }
    
  for (int i = 0; i < _outputs.size(); i++) {
    for (int j = 0; j < all_output_infos.size(); j++) {
      if (_outputs[i] == all_output_infos[j].name){
        _outputNodes.push_back(all_output_infos[j]);
      }
    }
  }
 
  if (_outputNodes.size() == 0){
    std::string s;
    for (const auto &piece : _outputs) {
      s += piece;
      s += " ";
    }
    throw EssentiaException(availableOutputInfo());
  }
    
  _isConfigured = true;
  reset();
    
  for (size_t i = 0; i < _nInputs; i++) {
    checkName(_inputs[i], all_input_infos);
  }
    
  for (size_t i = 0; i < _nOutputs; i++) {
    checkName(_outputs[i], all_output_infos);
  }
}

std::vector<TensorInfo> OnnxPredict::setTensorInfos(const Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator, bool is_input) {
    size_t count = is_input ? session.GetInputCount() : session.GetOutputCount();
    std::vector<TensorInfo> infos;
    
    auto names_raw = is_input
        ? session.GetInputNames()
        : session.GetOutputNames();

    for (size_t i = 0; i < count; ++i) {
        auto name_raw = names_raw[i];

        std::string name(name_raw);
        
        Ort::TypeInfo type_info = is_input
            ? session.GetInputTypeInfo(i)
            : session.GetOutputTypeInfo(i);

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
    std::cout << "=== " << label << " ===\n";
    for (const auto& info : infos) {
        std::cout << "[Name] " << info.name << endl;
        std::cout << "  [Type] " << info.type << endl;
        std::cout << "  [Shape] [";
        for (size_t j = 0; j < info.shape.size(); ++j) {
            std::cout << info.shape[j];
            if (j + 1 < info.shape.size()) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

std::string OnnxPredict::getTensorInfos(const std::vector<TensorInfo>& infos, const std::string& label) {
  std::string out;
  out += "=== " + label + " ===\n";
  for (const auto& info : infos) {
    out += "[Name] " + info.name + "\n";
    //cout << "info.type: " << typeid(info.type).name() << endl;
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
  if (!_isConfigured) return;
    
  input_names.clear();
  output_names.clear();
}

void OnnxPredict::compute() {
    
  if (!_isConfigured) {
    throw EssentiaException("OnnxPredict: This algorithm is not configured. To configure this algorithm you "
                            "should specify a valid `graphFilename` as input parameter.");
  }

  const Pool& poolIn = _poolIn.get();
  Pool& poolOut = _poolOut.get();
    
  std::vector<int64_t> shape;
    
  // Parse the input tensors from the pool into ONNX Runtime tensors.
  for (size_t i = 0; i < _nInputs; i++) {
    
    cout << "_inputs[i]: " << _inputs[i] << endl;
    const Tensor<Real>& inputData = poolIn.value<Tensor<Real> >(_inputs[i]);
      
    // Convert data to float32
    std::vector<float> float_data(inputData.size());
    for (size_t j = 0; j < inputData.size(); ++j) {
      float_data[j] = static_cast<float>(inputData.data()[j]);
    }
      
    // Step 2: Get shape
    int dims = 1;

    shape.push_back((int64_t)inputData.dimension(0));
    
    if (_squeeze) {
      for(int i = 1; i < inputData.rank(); i++) {
        if (inputData.dimension(i) > 1) {
          shape.push_back((int64_t)inputData.dimension(i));
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
      for(int j = 1; j < dims; j++) {   // HERE we need to jump i = 1 - 4D tensor input
          //cout << inputData.dimension(j) << endl;
        shape.push_back((int64_t)inputData.dimension(j));
      }
    }
            
    // Step 3: Create ONNX Runtime tensor
    _memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
      
    if (_memoryInfo == NULL) {
      throw EssentiaException("OnnxRuntimePredict: Error allocating memory for input tensor.");
    }

    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(_memoryInfo, float_data.data(), float_data.size(), shape.data(), shape.size()));

  }

  // Define input and output names
  for (const auto& tensorInfo : _inputNodes) {
    input_names.push_back(tensorInfo.name.c_str());
  }
    
  for (const auto& tensorInfo : _outputNodes) {
    output_names.push_back(tensorInfo.name.c_str());
  }
    
  // Run the Onnxruntime session.
  auto output_tensors = _session.Run(_runOptions,                     // Run options.
                                     input_names.data(),         // Input node names.
                                     input_tensors.data(),            // Input tensor values.
                                     _nInputs,                        // Number of inputs.
                                     output_names.data(),        // Output node names.
                                     _nOutputs                        // Number of outputs.
                                     );
    
  // Map output tensors to pool
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    
    const Real* outputData = output_tensors[i].GetTensorData<Real>();
    
    // Create and array to store the tensor shape.
    array<long int, 4> _shape {1, 1, 1, 1};
    //_shape[0] = (int)outputShapes[0];
    _shape[0] = (int)shape[0];
    for (size_t j = 1; j < _outputNodes[i].shape.size(); j++){
        _shape[j+1] = (int)_outputNodes[i].shape[j];
    }
    
    // Store tensor in pool
    const Tensor<Real> tensorMap = TensorMap<const Real>(outputData, _shape);
    poolOut.set(_outputs[i], tensorMap);
  }

  /* Cleanup
  for (const auto& tensorInfo : all_input_infos) {
    _allocator.Free((void*)tensorInfo.name.c_str());
  }
    
  for (const auto& tensorInfo : all_output_infos) {
    _allocator.Free((void*)tensorInfo.name.c_str());
  }*/
    
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
