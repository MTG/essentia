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

using namespace std;
using namespace essentia;
using namespace standard;

const char* TensorflowPredict::name = "TensorflowPredict";
const char* TensorflowPredict::category = "Machine Learning";
const char* TensorflowPredict::description = DOC("This algorithm returns a pool of output tensors generated form Tensorflow _graph.\n"
"This algorithm is a wrapper for the Tensorflow C API[1]."
"\n"
"References:\n"
"  [1] TensorFlow - An open source machine learning library for research and production.\n"
"  https://www.tensorflow.org/api_docs/cc/");


static void DeallocateBuffer(void* data, size_t) {
  // todo throws a warning. Why size_t?
  std::free(data);
}

void TensorflowPredict::configure() {
  string garphFilename = parameter("garphFilename").toString();
  _nameIn = parameter("namespaceIn").toString();
  _nameOut = parameter("namespaceOut").toString();
  // _fetchOutputs = parameter("fetchOutputs").toVectorString();
  _nOutputs = _fetchOutputs.size();

  // First we load and initialize the model.
  // Code inspired in https://github.com/Neargye/hello_tf_c_api/,
  // rather cite or refactor!
  const auto f = fopen(garphFilename.c_str(), "rb");
  if (f == nullptr) {
    throw(EssentiaException("TensorflowPredict: could not open the _graph file"));
  }

  fseek(f, 0, SEEK_END);
  const auto fsize = ftell(f);
  fseek(f, 0, SEEK_SET);

  // size sanity check
  if (fsize < 1) {
    fclose(f);
    throw(EssentiaException("TensorflowPredict: _graph file is empty"));
  }

  // reserve memory and read the _graph as binary data
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
    throw(EssentiaException("TensorflowPredict: Graph status is ", _status));
  }

  _sessionOptions = TF_NewSessionOptions();
  _session = TF_NewSession(_graph, _sessionOptions, _status);

  // where to delete?
  TF_DeleteSessionOptions(_sessionOptions);
  
  if (TF_GetCode(_status) != TF_OK) {
    throw(EssentiaException("TensorflowPredict: Session status is ", _status));
  }
}


void TensorflowPredict::compute() {

  const Pool& poolIn = _poolIn.get();
  Pool& poolOut = _poolOut.get();

  // get data from the pool

  string nameOut = parameter("namespaceOut").toString();
  vector<vector<Real> > rawFeats = poolIn.value<vector<vector<Real> > >(_nameIn);

  // TODO pool to tensors and tensors to pool again un to 3 or 4D 
  //  NdArray<Real, 4> input_tensors();
  //  NdArray<Real, 4> output_tensors();

  /*  
  // Run the session
  TF_SessionRun(_session,
                nullptr, // Run options.
                inputs, &input_tensors[0][0][0][0], 1, // Input tensors, input tensor values, number of inputs.
                outputs, &output_tensors[0][0][0][0], static_cast<int>(_nOutputs), // Output tensors, output tensor values, number of outputs.
                nullptr, 0, // Target operations, number of targets.
                nullptr, // Run metadata.
                _status // Output status. 
);
  */


  // poolOut.add(nameOut, output);
}

