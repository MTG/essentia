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

#ifndef ESSENTIA_STREAMING_TENSORFLOWPREDICTMUSICNN_H
#define ESSENTIA_STREAMING_TENSORFLOWPREDICTMUSICNN_H


#include "streamingalgorithmcomposite.h"
#include "algorithmfactory.h"
#include "algorithm.h"
#include "network.h"

namespace essentia {
namespace streaming {

class TensorflowPredictMusiCNN : public AlgorithmComposite {
 protected:
  Algorithm* _frameCutter;
  Algorithm* _tensorflowInputMusiCNN;
  Algorithm* _vectorRealToTensor;
  Algorithm* _tensorToPool;
  Algorithm* _tensorflowPredict;
  Algorithm* _poolToTensor;
  Algorithm* _tensorToVectorReal;

  SinkProxy<Real> _signal;
  SourceProxy<std::vector<Real> > _predictions;

  scheduler::Network* _network;
  bool _configured;

  void createInnerNetwork();
  void clearAlgos();

 public:
  TensorflowPredictMusiCNN();
  ~TensorflowPredictMusiCNN();

  void declareParameters() {
    declareParameter("graphFilename", "the name of the file containing the model to use", "", Parameter::STRING);
    declareParameter("input", "the name of the input nodes in the Tensorflow graph", "", "model/Placeholder");
    declareParameter("output", "the name of the node from which to retrieve the output tensors", "", "model/Sigmoid");
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_frameCutter));
  }

  void configure();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#include "vectorinput.h"
#include "pool.h"
#include "poolstorage.h"

namespace essentia {
namespace standard {

// Standard non-streaming algorithm comes after the streaming one as it
// depends on it
class TensorflowPredictMusiCNN : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<std::vector<Real> > > _predictions;

  streaming::Algorithm* _tensorflowPredictMusiCNN;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

  void createInnerNetwork();

 public:
  TensorflowPredictMusiCNN();
  ~TensorflowPredictMusiCNN();

  void declareParameters() {
    declareParameter("graphFilename", "the name of the file containing the model to use", "", Parameter::STRING);
    declareParameter("input", "the name of the input nodes in the Tensorflow graph", "", "model/Placeholder");
    declareParameter("output", "the name of the node from which to retrieve the output tensors", "", "model/Sigmoid");
  }

  void configure();
  void compute();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif // ESSENTIA_STREAMING_TENSORFLOWPREDICTMUSICNN_H
