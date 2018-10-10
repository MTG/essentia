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

  std::string _nameIn;
  std::string _nameOut;
  std::vector<std::string> _fetchOutputs;
  int _nOutputs;

  TF_Graph* _graph;
  TF_Status* _status;
  TF_ImportGraphDefOptions* _options;
  TF_SessionOptions* _sessionOptions;
  TF_Session* _session;


 public:
  TensorflowPredict() {
    declareInput(_poolIn, "poolIn", "the pool where to get the feature tensors");
    declareOutput(_poolOut, "poolOut", "the pool where to store the predicted tensors");
  }

  ~TensorflowPredict(){
    TF_DeleteImportGraphDefOptions(_options);
    TF_DeleteStatus(_status);
    TF_DeleteGraph(_graph);
  }

  void declareParameters() {
    declareParameter("garphFilename", "the name of the file from which to read the Tensorflow graph", "", "/home/pablo/base_model.pb");
    declareParameter("fetchOutputs", "will save the output tensors of the graph nodes named after each element of this vector of strings","", Parameter::VECTOR_STRING);
    declareParameter("namespaceIn", "will look for this namespace in poolIn", "", "X");
    declareParameter("namespaceOut", "will save to this namespace in poolOut", "", "Y");
  }

  void configure();
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;

};

} //namespace standard
} //namespace essentia

#endif // ESSENTIA_TENSORFLOWPREDICT_H
